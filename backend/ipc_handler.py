"""
Production-ready async IPC handler for Python backend.
Fixes critical issues with stdin/stdout async I/O and adds proper error handling.
"""
import sys
import json
import asyncio
import logging
import traceback
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


class IPCHandler:
    """
    Handles JSON-RPC communication over stdin/stdout with proper async I/O.
    
    Key improvements:
    - Proper async I/O using run_in_executor
    - Request queuing and concurrency control
    - Timeout handling
    - Graceful shutdown
    - Comprehensive error handling
    """
    
    def __init__(self, max_concurrent_requests: int = 5, request_timeout: float = 30.0):
        """
        Initialize IPC handler.
        
        Args:
            max_concurrent_requests: Maximum number of concurrent requests to process
            request_timeout: Timeout in seconds for each request
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.running = False
        self.handlers: Dict[str, Callable] = {}
        self.active_requests = 0
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"IPCHandler initialized with max_concurrent={max_concurrent_requests}, "
                   f"timeout={request_timeout}s")
    
    def register_handler(self, command: str, handler: Callable):
        """Register a command handler."""
        self.handlers[command] = handler
        logger.info(f"Registered handler for command: {command}")
    
    async def start(self):
        """Start the IPC loop."""
        self.running = True
        logger.info("Starting IPC handler...")
        
        # Set up signal handlers for graceful shutdown (Unix only)
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            try:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                # Fallback for platforms/loops that don't support signal handlers
                logger.warning("Signal handlers not supported on this platform/event loop")
        
        try:
            await self._run_loop()
        except Exception as e:
            logger.error(f"Fatal error in IPC loop: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def _run_loop(self):
        """Main IPC loop with proper async I/O."""
        loop = asyncio.get_event_loop()
        
        logger.info("IPC loop started, ready for commands")
        print("READY", file=sys.stderr, flush=True)  # Signal to Rust that we're ready
        
        while self.running:
            try:
                # Read line from stdin asynchronously
                line = await loop.run_in_executor(None, self._read_line)
                
                if line is None:  # EOF
                    logger.info("EOF received, shutting down")
                    break
                
                if not line.strip():
                    continue
                
                # Process request asynchronously with concurrency control
                asyncio.create_task(self._process_request(line))
                
            except Exception as e:
                logger.error(f"Error reading from stdin: {e}", exc_info=True)
                # Don't break - try to continue
                await asyncio.sleep(0.1)
    
    def _read_line(self) -> Optional[str]:
        """
        Read a line from stdin (blocking operation for executor).
        
        Returns:
            Line string or None on EOF
        """
        try:
            line = sys.stdin.readline()
            if not line:  # EOF
                return None
            return line.strip()
        except Exception as e:
            logger.error(f"Error reading stdin: {e}")
            return None
    
    async def _process_request(self, line: str):
        """
        Process a single JSON-RPC request with timeout and error handling.
        
        Args:
            line: JSON-RPC request string
        """
        request_id = None
        start_time = datetime.now()
        
        # Acquire semaphore for concurrency control
        async with self.request_semaphore:
            self.active_requests += 1
            
            try:
                # Parse JSON-RPC request
                try:
                    request = json.loads(line)
                    request_id = request.get("id")
                    method = request.get("method")
                    params = request.get("params", {})
                    
                    logger.info(f"Processing request {request_id}: {method}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    await self._send_error(
                        request_id=None,
                        code=-32700,
                        message=f"Parse error: {str(e)}"
                    )
                    return
                
                # Validate request
                if not method:
                    await self._send_error(
                        request_id=request_id,
                        code=-32600,
                        message="Invalid request: missing method"
                    )
                    return
                
                # Get handler
                handler = self.handlers.get(method)
                if not handler:
                    await self._send_error(
                        request_id=request_id,
                        code=-32601,
                        message=f"Method not found: {method}"
                    )
                    return
                
                # Execute handler with timeout
                try:
                    result = await asyncio.wait_for(
                        handler(**params),
                        timeout=self.request_timeout
                    )
                    
                    # Send success response
                    await self._send_success(request_id, result)
                    
                    # Log execution time
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Request {request_id} completed in {execution_time:.3f}s")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Request {request_id} timed out after {self.request_timeout}s")
                    await self._send_error(
                        request_id=request_id,
                        code=-32000,
                        message=f"Request timeout after {self.request_timeout}s"
                    )
                
                except Exception as e:
                    logger.error(f"Handler error for {method}: {e}", exc_info=True)
                    await self._send_error(
                        request_id=request_id,
                        code=-32603,
                        message=f"Internal error: {str(e)}",
                        data={"traceback": traceback.format_exc()}
                    )
                    
            finally:
                self.active_requests -= 1
    
    async def _send_success(self, request_id: Any, result: Any):
        """Send a success response."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
            "error": None
        }
        await self._write_response(response)
    
    async def _send_error(self, request_id: Any, code: int, message: str, data: Any = None):
        """Send an error response."""
        error = {
            "code": code,
            "message": message
        }
        if data:
            error["data"] = data
        
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": None,
            "error": error
        }
        await self._write_response(response)
    
    async def _write_response(self, response: Dict[str, Any]):
        """
        Write response to stdout asynchronously.
        
        Args:
            response: JSON-RPC response dictionary
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Serialize to JSON
            response_str = json.dumps(response)
            
            # Write to stdout in executor (blocking I/O)
            await loop.run_in_executor(None, self._write_line, response_str)
            
        except Exception as e:
            logger.error(f"Failed to write response: {e}", exc_info=True)
    
    def _write_line(self, line: str):
        """Write a line to stdout (blocking operation for executor)."""
        try:
            print(line, file=sys.stdout, flush=True)
        except Exception as e:
            logger.error(f"Error writing to stdout: {e}")
    
    async def shutdown(self):
        """Graceful shutdown."""
        if not self.running:
            return
        
        logger.info("Shutting down IPC handler...")
        self.running = False
        
        # Wait for active requests to complete (with timeout)
        shutdown_timeout = 10.0
        wait_start = datetime.now()
        
        while self.active_requests > 0:
            if (datetime.now() - wait_start).total_seconds() > shutdown_timeout:
                logger.warning(f"Shutdown timeout: {self.active_requests} requests still active")
                break
            
            logger.info(f"Waiting for {self.active_requests} active requests to complete...")
            await asyncio.sleep(0.5)
        
        logger.info("IPC handler shutdown complete")
        self.shutdown_event.set()


# Convenience function for creating and running IPC handler
async def run_ipc_server(handlers: Dict[str, Callable], **kwargs):
    """
    Create and run IPC server.
    
    Args:
        handlers: Dictionary mapping command names to async handler functions
        **kwargs: Additional arguments for IPCHandler
    
    Example:
        async def handle_ping():
            return {"status": "ok"}
        
        await run_ipc_server({"ping": handle_ping})
    """
    ipc = IPCHandler(**kwargs)
    
    # Register all handlers
    for command, handler in handlers.items():
        ipc.register_handler(command, handler)
    
    # Start server
    await ipc.start()


if __name__ == "__main__":
    # Test the IPC handler
    async def test_handler(message: str = ""):
        """Test handler that echoes back."""
        await asyncio.sleep(0.1)  # Simulate work
        return {"echo": message, "timestamp": datetime.now().isoformat()}
    
    async def main():
        handlers = {
            "test": test_handler,
            "ping": lambda: {"status": "ok"}
        }
        await run_ipc_server(handlers)
    
    asyncio.run(main())