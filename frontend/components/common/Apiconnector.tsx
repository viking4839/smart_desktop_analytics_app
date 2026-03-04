import React, { useState } from 'react';
import { Sparkles, X, Send, Key, Loader, MessageCircle } from 'lucide-react';

export const AiAssistantWidget: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);

    // API State
    const [apiKey, setApiKey] = useState('');
    const [prompt, setPrompt] = useState('');
    const [aiResponse, setAiResponse] = useState('Hello! Enter your API key and ask me a question about your data.');
    const [loading, setLoading] = useState(false);

    const askGemini = async () => {
        // 1. Clean the key to remove accidental spaces from copy-pasting
        const cleanKey = apiKey.trim();

        if (!cleanKey) {
            setAiResponse("⚠️ Please enter your Gemini API key below.");
            return;
        }
        if (!prompt.trim()) return;

        setLoading(true);
        setAiResponse('');

        try {
            // Update your URL to look exactly like this:
            const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${cleanKey}`;

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contents: [
                        {
                            parts: [
                                { text: prompt }
                            ]
                        }
                    ]
                })
            });

            // 2. THE DIAGNOSTIC TOOL: If it fails, ask Google EXACTLY why
            if (!response.ok) {
                const errorText = await response.text(); // Read Google's exact complaint
                console.error("Raw Google Error:", errorText);
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            const answer = data.candidates[0].content.parts[0].text;
            setAiResponse(answer);
            setPrompt('');

        } catch (error: any) {
            console.error("Full Error Object:", error);
            // 3. Print the exact error to the UI so you can read it
            setAiResponse(`❌ Connection Failed.\n\nDetails: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };
    return (
        <div style={{ position: 'fixed', bottom: '24px', right: '24px', zIndex: 9999, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '16px' }}>

            {/* THE POPUP CHAT PANEL */}
            {isOpen && (
                <div style={{
                    width: '350px',
                    height: '500px',
                    background: '#ffffff',
                    borderRadius: '12px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.15)',
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden',
                    border: '1px solid #e2e8f0'
                }}>
                    {/* Header */}
                    <div style={{ background: '#0f172a', padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'white' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                            <Sparkles size={18} color="#38bdf8" /> Data Assistant
                        </div>
                        <button onClick={() => setIsOpen(false)} style={{ background: 'none', border: 'none', color: '#cbd5e1', cursor: 'pointer' }}>
                            <X size={18} />
                        </button>
                    </div>

                    {/* Chat History / AI Response Area */}
                    <div style={{ flex: 1, padding: '16px', overflowY: 'auto', background: '#f8fafc', fontSize: '14px', lineHeight: '1.6', color: '#334155', whiteSpace: 'pre-wrap' }}>
                        {loading ? (
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', color: '#64748b' }}>
                                <Loader size={16} className="spinner" /> AI is thinking...
                            </div>
                        ) : (
                            aiResponse
                        )}
                    </div>

                    {/* Input Area (Bottom of Panel) */}
                    <div style={{ padding: '16px', borderTop: '1px solid #e2e8f0', background: 'white', display: 'flex', flexDirection: 'column', gap: '12px' }}>

                        {/* API Key Input */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px', background: '#f1f5f9', borderRadius: '6px', border: '1px solid #cbd5e1' }}>
                            <Key size={14} color="#64748b" />
                            <input
                                type="password"
                                placeholder="Paste your Gemini API Key..."
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', fontSize: '12px' }}
                            />
                        </div>

                        {/* Prompt Input & Send Button */}
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <input
                                type="text"
                                placeholder="Ask a question..."
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && askGemini()} // Send on Enter key
                                style={{ flex: 1, padding: '10px', borderRadius: '6px', border: '1px solid #cbd5e1', outline: 'none' }}
                            />
                            <button
                                onClick={askGemini}
                                disabled={loading || !prompt.trim()}
                                style={{ padding: '10px 14px', background: '#0ea5e9', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                            >
                                <Send size={16} />
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* THE FLOATING ACTION BUTTON */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                style={{
                    width: '60px',
                    height: '60px',
                    borderRadius: '30px',
                    background: '#0f172a',
                    color: 'white',
                    border: 'none',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'transform 0.2s',
                    transform: isOpen ? 'scale(0.9)' : 'scale(1)'
                }}
            >
                {isOpen ? <X size={28} /> : <MessageCircle size={28} />}
            </button>
        </div>
    );
};