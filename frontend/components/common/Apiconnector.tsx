import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Sparkles, X, Send, Key, Loader, BarChart3, Lightbulb,
    SearchCheck, TrendingUp, FileText, Settings, AlertTriangle,
    Trash2, Copy, Check
} from 'lucide-react';

// ─────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────
interface ColumnInfo {
    name: string;
    data_type: string;
    nullable?: boolean;
    unique_values?: number;
    sample_values?: any[];
    statistics?: {
        numeric_stats?: {
            min: number; max: number; mean: number; median: number;
            std: number; q25: number; q75: number;
            sum?: number;           // total sum of non-null values
            non_null_count?: number; // rows that have a value
        };
        categorical_stats?: { top_categories: Record<string, number>; category_count: number; };
    };
}

interface QualityScore {
    score: number;
    completeness: number;
    null_percentage: number;
    issues?: string[];
}

interface QueryResult {
    summary?: Record<string, any>;
    statement?: string;
    table?: { columns: string[]; rows: any[][]; row_count: number; };
    chart?: { type: string; title: string; x_axis: string; y_axis: string; data: any[]; };
    provenance?: { datasets: string[]; columns: string[]; operations: string[]; execution_time: number; };
    success?: boolean;
}

interface Dataset {
    id: string;
    name: string;
    row_count: number;
    column_count: number;
    schema?: Record<string, ColumnInfo>;
}

export interface AiAssistantWidgetProps {
    datasets: Dataset[];
    selectedDatasetId: string | null;
    currentResult: QueryResult | null;
    // Rich props — already computed in App.tsx, zero extra backend cost
    availableColumns?: ColumnInfo[];
    quickInsights?: string[];
    suggestedQueries?: any[];
    qualityScore?: QualityScore | null;
}

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
}

// ─────────────────────────────────────────────
//  Token budget constants
//  Target: < 2 000 tokens per request
//  System ~400 | Context ~600 | History ~600 | User ~100
// ─────────────────────────────────────────────
const MAX_HISTORY_TURNS = 4;   // last 4 exchanges = 8 messages
const MAX_SAMPLE_ROWS = 5;
const MAX_COLS_SCHEMA = 25;

// ─────────────────────────────────────────────
//  Intent Classifier
//  Routes question to the right context tier
// ─────────────────────────────────────────────
type PromptIntent = 'quality' | 'schema' | 'result' | 'suggest' | 'aggregate' | 'general';

function classifyIntent(text: string): PromptIntent {
    const t = text.toLowerCase();
    if (/null|missing|quality|issue|clean|duplicate|valid|completeness/.test(t)) return 'quality';
    if (/\bsum\b|total|count|how many|average|mean|median|min|max|aggr|group by|breakdown/.test(t)) return 'aggregate';
    if (/column|schema|field|type|structure|describe|what.*data|variable/.test(t)) return 'schema';
    if (/result|chart|trend|takeaway|explain|summary|analys|insight|top/.test(t)) return 'result';
    if (/suggest|recommend|what.*query|how.*analys|what.*look/.test(t)) return 'suggest';
    return 'general';
}

// ─────────────────────────────────────────────
//  Tiered Context Builder
//  Only attaches what the intent actually needs.
// ─────────────────────────────────────────────
function buildContext(
    intent: PromptIntent,
    dataset: Dataset | undefined,
    availableColumns: ColumnInfo[],
    quickInsights: string[],
    suggestedQueries: any[],
    qualityScore: QualityScore | null | undefined,
    currentResult: QueryResult | null
): string {
    if (!dataset) return 'No dataset currently selected.';

    const parts: string[] = [];

    // TIER 0 — always (dataset identity, ~25 tokens)
    parts.push(`Dataset: "${dataset.name}" | ${dataset.row_count.toLocaleString()} rows × ${dataset.column_count} cols`);

    // TIER 1 — schema (schema / suggest / general / aggregate intents)
    if (['schema', 'suggest', 'general', 'aggregate'].includes(intent) && availableColumns.length > 0) {
        const cols = availableColumns.slice(0, MAX_COLS_SCHEMA).map(c => {
            if (c.statistics?.numeric_stats) {
                const s = c.statistics.numeric_stats;
                const count = s.non_null_count ?? dataset.row_count;
                // Always include sum so the AI can answer aggregate questions directly
                const sumStr = s.sum !== undefined
                    ? `, sum=${s.sum.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
                    : `, sum≈${(s.mean * count).toLocaleString(undefined, { maximumFractionDigits: 2 })}(derived)`;
                return `${c.name}(${c.data_type})[min=${s.min}, max=${s.max}, mean=${s.mean.toFixed(2)}, count=${count}${sumStr}]`;
            }
            if (c.statistics?.categorical_stats) {
                const s = c.statistics.categorical_stats;
                const topEntries = Object.entries(s.top_categories).slice(0, 3)
                    .map(([k, v]) => `"${k}":${v}`).join(', ');
                return `${c.name}(${c.data_type})[${s.category_count} unique, top: ${topEntries}]`;
            }
            return `${c.name}(${c.data_type})`;
        });
        parts.push(`Columns: ${cols.join(' | ')}`);
        if (availableColumns.length > MAX_COLS_SCHEMA)
            parts.push(`(+${availableColumns.length - MAX_COLS_SCHEMA} more columns)`);
    }

    // TIER 2 — quality (quality / general intents)
    if (['quality', 'general'].includes(intent) && qualityScore) {
        parts.push(`Data Quality: ${qualityScore.score}/100 | Completeness: ${qualityScore.completeness}% | Null rate: ${qualityScore.null_percentage}%`);
        if (qualityScore.issues?.length)
            parts.push(`Issues: ${qualityScore.issues.slice(0, 3).join('; ')}`);
    }

    // TIER 3 — Python quick insights (suggest / general)
    if (['suggest', 'general'].includes(intent) && quickInsights.length > 0)
        parts.push(`Pre-computed insights: ${quickInsights.slice(0, 4).join(' | ')}`);

    // TIER 4 — suggested queries (suggest only)
    if (intent === 'suggest' && suggestedQueries.length > 0) {
        const names = suggestedQueries.slice(0, 4).map((q: any) => q.name || q.description).join(', ');
        parts.push(`Query templates available: ${names}`);
    }

    // TIER 5 — query result (result / general / aggregate)
    if (['result', 'general', 'aggregate'].includes(intent) && currentResult) {
        if (currentResult.statement)
            parts.push(`Last query: "${currentResult.statement}"`);
        if (currentResult.table) {
            const t = currentResult.table;
            const sample = t.rows.slice(0, MAX_SAMPLE_ROWS).map(row =>
                Object.fromEntries(t.columns.map((c, i) => [c, row[i]]))
            );
            parts.push(`Result: ${t.row_count} rows | Cols: [${t.columns.join(', ')}] | Sample: ${JSON.stringify(sample)}`);
        }
        if (currentResult.provenance) {
            const p = currentResult.provenance;
            parts.push(`Ops: ${p.operations.join('→')} | ${p.execution_time.toFixed(3)}s`);
        }
        if (currentResult.chart)
            parts.push(`Chart: ${currentResult.chart.type} — "${currentResult.chart.title}" (X:${currentResult.chart.x_axis} Y:${currentResult.chart.y_axis})`);
    }

    return parts.join('\n');
}

// ─────────────────────────────────────────────
//  System Prompt
// ─────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a senior data analyst assistant in a desktop analytics app.
Rules:
- Be concise and specific — cite actual column names, numbers, and percentages from context.
- For AGGREGATE questions (sum, count, mean, min, max, median): the column stats in context contain pre-computed values — use them directly to answer. Each numeric column is provided with min, max, mean, count (non-null rows), and sum. Do NOT say you cannot compute aggregates; the values are already in the context.
- If a sum is marked "(derived)" it was calculated as mean × count and may differ slightly from the exact sum if nulls exist — note this briefly.
- When suggesting next steps, reference columns or operations that exist in the dataset.
- Format numbers with commas. Use bullet points for 3+ items.
- Never hallucinate column names or statistics not in the context.
- If context is truly insufficient (e.g. a filtered subset not in context), say so and suggest running the query in the app.
- Skip filler phrases like "Great question!" or "Certainly!".`;

// ─────────────────────────────────────────────
//  Dynamic Suggestions (context-aware)
// ─────────────────────────────────────────────
function buildSuggestions(
    dataset: Dataset | undefined,
    currentResult: QueryResult | null,
    qualityScore: QualityScore | null | undefined,
    cols: ColumnInfo[]
): Array<{ text: string; icon: React.ReactNode; intent: PromptIntent }> {
    if (currentResult?.table) return [
        { text: 'Explain these results', icon: <FileText size={12} />, intent: 'result' },
        { text: 'Find trends & patterns', icon: <TrendingUp size={12} />, intent: 'result' },
        { text: 'What to investigate next?', icon: <Lightbulb size={12} />, intent: 'suggest' },
    ];
    if (dataset) {
        const num = cols.find(c => ['number', 'integer', 'float'].includes(c.data_type));
        const cat = cols.find(c => ['string', 'object'].includes(c.data_type));
        const out: Array<{ text: string; icon: React.ReactNode; intent: PromptIntent }> = [];
        if (num) out.push({ text: `Analyse ${num.name} distribution`, icon: <BarChart3 size={12} />, intent: 'schema' });
        if (cat) out.push({ text: `Top values in ${cat.name}`, icon: <SearchCheck size={12} />, intent: 'result' });
        if (qualityScore && qualityScore.score < 90)
            out.push({ text: `Data quality issues (${qualityScore.score}/100)`, icon: <AlertTriangle size={12} />, intent: 'quality' });
        else
            out.push({ text: 'Suggest queries for this data', icon: <Lightbulb size={12} />, intent: 'suggest' });
        return out;
    }
    return [{ text: 'How do I get started?', icon: <Lightbulb size={12} />, intent: 'general' }];
}

// ─────────────────────────────────────────────
//  Markdown-lite renderer (bold + bullets)
// ─────────────────────────────────────────────
function renderMarkdown(text: string): React.ReactNode {
    return text.split('\n').map((line, i) => {
        const bullet = line.match(/^[-*•]\s+(.*)/);
        if (bullet) return (
            <div key={i} style={{ display: 'flex', gap: '6px', marginBottom: '2px' }}>
                <span style={{ color: '#667eea', fontWeight: 700, flexShrink: 0 }}>•</span>
                <span>{renderInline(bullet[1])}</span>
            </div>
        );
        if (!line.trim()) return <div key={i} style={{ height: '6px' }} />;
        return <div key={i} style={{ marginBottom: '2px' }}>{renderInline(line)}</div>;
    });
}
function renderInline(text: string): React.ReactNode {
    return text.split(/(\*\*[^*]+\*\*)/g).map((p, i) =>
        p.startsWith('**') && p.endsWith('**') ? <strong key={i}>{p.slice(2, -2)}</strong> : p
    );
}

// ─────────────────────────────────────────────
//  Component
// ─────────────────────────────────────────────
export const AiAssistantWidget: React.FC<AiAssistantWidgetProps> = ({
    datasets,
    selectedDatasetId,
    currentResult,
    availableColumns = [],
    quickInsights = [],
    suggestedQueries = [],
    qualityScore,
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const [apiKey, setApiKey] = useState(() => localStorage.getItem('gemini_api_key') || '');
    const [showKeySettings, setShowKeySettings] = useState(() => !localStorage.getItem('gemini_api_key'));
    const [prompt, setPrompt] = useState('');
    const [streamedText, setStreamedText] = useState('');
    const [historyMap, setHistoryMap] = useState<Record<string, ChatMessage[]>>(() => {
        try {
            const saved = localStorage.getItem('smart_ai_chat_history');
            return saved ? JSON.parse(saved) : {};
        } catch { return {}; }
    });
    const activeId = selectedDatasetId || 'global';
    const messages = historyMap[activeId] || [];
    const setMessages = useCallback((action: React.SetStateAction<ChatMessage[]>) => {
        setHistoryMap(prevMap => {
            const currentMsgs = prevMap[activeId] || [];
            const nextMsgs = typeof action === 'function' ? action(currentMsgs) : action;

            const newMap = { ...prevMap, [activeId]: nextMsgs };
            // Save to memory so it survives app restarts
            localStorage.setItem('smart_ai_chat_history', JSON.stringify(newMap));
            return newMap;
        });
    }, [activeId]);
    const [loading, setLoading] = useState(false);
    const [copied, setCopied] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const activeDataset = datasets.find(d => d.id === selectedDatasetId);

    useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, streamedText]);
    useEffect(() => { if (isOpen) setTimeout(() => inputRef.current?.focus(), 150); }, [isOpen]);
    useEffect(() => { setStreamedText(''); }, [selectedDatasetId]);

    const suggestions = buildSuggestions(activeDataset, currentResult, qualityScore, availableColumns);

    // ─────────────────────────────────────────
    //  Core AI call — Gemini SSE streaming
    // ─────────────────────────────────────────
    const askGemini = useCallback(async (overridePrompt?: string, overrideIntent?: PromptIntent) => {
        const text = (overridePrompt || prompt).trim();
        const cleanKey = apiKey.trim();

        if (!cleanKey) {
            setMessages(p => [...p, { role: 'assistant', content: '⚠️ Please enter your Gemini API key (gear icon).', timestamp: Date.now() }]);
            setShowKeySettings(true);
            return;
        }
        if (!text) return;

        localStorage.setItem('gemini_api_key', cleanKey);
        setShowKeySettings(false);
        setPrompt('');
        setLoading(true);
        setStreamedText('');

        const userMsg: ChatMessage = { role: 'user', content: text, timestamp: Date.now() };
        const updatedMsgs = [...messages, userMsg];
        setMessages(updatedMsgs);

        // Build tiered context
        const intent = overrideIntent || classifyIntent(text);
        const ctxBlock = buildContext(intent, activeDataset, availableColumns, quickInsights, suggestedQueries, qualityScore, currentResult);

        // Trim history to token budget
        const historySlice = updatedMsgs.slice(-(MAX_HISTORY_TURNS * 2), -1);
        const geminiHistory = historySlice.map(m => ({
            role: m.role === 'user' ? 'user' : 'model',
            parts: [{ text: m.content }]
        }));

        // System context injected as a framing user/model exchange
        const contents = [
            { role: 'user', parts: [{ text: `${SYSTEM_PROMPT}\n\n--- DATA CONTEXT ---\n${ctxBlock}\n--- END CONTEXT ---` }] },
            { role: 'model', parts: [{ text: 'Understood. I have the dataset context. Ready.' }] },
            ...geminiHistory,
            { role: 'user', parts: [{ text }] }
        ];

        try {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=${cleanKey}`;
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents,
                    generationConfig: { temperature: 0.4, maxOutputTokens: 2000, topP: 0.8 },
                })
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

            const reader = res.body!.getReader();
            const decoder = new TextDecoder();
            let fullText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const lines = decoder.decode(value, { stream: true }).split('\n').filter(l => l.startsWith('data: '));
                for (const line of lines) {
                    try {
                        const json = JSON.parse(line.slice(6));
                        const piece = json.candidates?.[0]?.content?.parts?.[0]?.text || '';
                        fullText += piece;
                        setStreamedText(fullText);
                    } catch { /* partial chunk, skip */ }
                }
            }

            setMessages(p => [...p, { role: 'assistant', content: fullText || 'No response received.', timestamp: Date.now() }]);
            setStreamedText('');

        } catch (err: any) {
            const msg = err.message?.includes('API_KEY_INVALID')
                ? '❌ Invalid API key — please check your Gemini key in settings.'
                : `❌ Request failed: ${err.message}`;
            setMessages(p => [...p, { role: 'assistant', content: msg, timestamp: Date.now() }]);
            setStreamedText('');
        } finally {
            setLoading(false);
        }
    }, [prompt, apiKey, messages, activeDataset, availableColumns, quickInsights, suggestedQueries, qualityScore, currentResult]);

    const copyLast = () => {
        const last = [...messages].reverse().find(m => m.role === 'assistant');
        if (last) { navigator.clipboard.writeText(last.content); setCopied(true); setTimeout(() => setCopied(false), 2000); }
    };

    const ctxBadge = activeDataset ? { label: activeDataset.name, dot: '#10b981' } : { label: 'No dataset', dot: '#94a3b8' };

    // ──────────── Shared button style helpers ────────────
    const iconBtn = (title: string, onClick: () => void, children: React.ReactNode) => (
        <button onClick={onClick} title={title} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer', opacity: 0.75, padding: '2px', display: 'flex', alignItems: 'center' }}>
            {children}
        </button>
    );

    return (
        <div style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 9999, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 16 }}>

            {isOpen && (
                <div style={{ width: 400, height: 610, background: '#fff', borderRadius: 16, boxShadow: '0 12px 40px rgba(0,0,0,0.18)', display: 'flex', flexDirection: 'column', overflow: 'hidden', border: '1px solid #e2e8f0' }}>

                    {/* ── Header ── */}
                    <div style={{ background: 'linear-gradient(135deg,#667eea,#764ba2)', padding: '13px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'white', flexShrink: 0 }}>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontWeight: 700, fontSize: 15 }}>
                                <Sparkles size={15} /> Smart Analyst
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, opacity: 0.88, marginTop: 2 }}>
                                <div style={{ width: 6, height: 6, borderRadius: '50%', background: ctxBadge.dot, flexShrink: 0 }} />
                                <span style={{ maxWidth: 230, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {ctxBadge.label}
                                    {activeDataset && ` · ${activeDataset.row_count.toLocaleString()} rows`}
                                    {qualityScore && ` · Q:${qualityScore.score}/100`}
                                </span>
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                            {messages.length > 0 && iconBtn('Copy last response', copyLast, copied ? <Check size={15} /> : <Copy size={15} />)}
                            {messages.length > 0 && iconBtn('Clear chat', () => setMessages([]), <Trash2 size={15} />)}
                            {iconBtn('API key settings', () => setShowKeySettings(s => !s), <Settings size={15} />)}
                            {iconBtn('Close', () => setIsOpen(false), <X size={17} />)}
                        </div>
                    </div>

                    {/* ── API Key field ── */}
                    {showKeySettings && (
                        <div style={{ padding: '9px 14px', background: '#fef3c7', borderBottom: '1px solid #fde68a', display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                            <Key size={13} color="#d97706" style={{ flexShrink: 0 }} />
                            <input
                                type="password"
                                placeholder="Paste your Gemini API key…"
                                value={apiKey}
                                onChange={e => setApiKey(e.target.value)}
                                onBlur={() => { if (apiKey.trim()) { localStorage.setItem('gemini_api_key', apiKey.trim()); setShowKeySettings(false); } }}
                                style={{ border: 'none', background: 'transparent', outline: 'none', flex: 1, fontSize: 12, fontFamily: 'monospace' }}
                            />
                        </div>
                    )}

                    {/* ── Message History ── */}
                    <div style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 10, background: '#f8fafc' }}>

                        {messages.length === 0 && !streamedText && (
                            <div style={{ textAlign: 'center', padding: '28px 16px', color: '#94a3b8' }}>
                                <Sparkles size={26} style={{ margin: '0 auto 10px', display: 'block', opacity: 0.35 }} />
                                <div style={{ fontSize: 14, fontWeight: 600, color: '#64748b', marginBottom: 5 }}>
                                    {activeDataset ? `Ready to analyse "${activeDataset.name}"` : 'Select a dataset to begin'}
                                </div>
                                <div style={{ fontSize: 12 }}>
                                    {activeDataset
                                        ? 'Ask anything or tap a suggestion below. I have access to the full schema and statistics.'
                                        : 'Once a dataset is loaded I can help you explore, analyse, and interpret it.'}
                                </div>
                            </div>
                        )}

                        {messages.map((msg, i) => (
                            <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                                <div style={{
                                    maxWidth: '88%', padding: '9px 13px',
                                    borderRadius: msg.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                                    background: msg.role === 'user' ? 'linear-gradient(135deg,#667eea,#764ba2)' : 'white',
                                    color: msg.role === 'user' ? 'white' : '#1e293b',
                                    fontSize: 13.5, lineHeight: 1.55,
                                    boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
                                    border: msg.role === 'assistant' ? '1px solid #e2e8f0' : 'none',
                                }}>
                                    {msg.role === 'assistant' ? renderMarkdown(msg.content) : msg.content}
                                </div>
                            </div>
                        ))}

                        {/* Live streaming bubble */}
                        {streamedText && (
                            <div style={{ display: 'flex' }}>
                                <div style={{ maxWidth: '88%', padding: '9px 13px', borderRadius: '14px 14px 14px 4px', background: 'white', color: '#1e293b', fontSize: 13.5, lineHeight: 1.55, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', border: '1px solid #e2e8f0' }}>
                                    {renderMarkdown(streamedText)}
                                    <span style={{ display: 'inline-block', width: 7, height: 13, background: '#667eea', marginLeft: 2, borderRadius: 1, animation: 'blink 1s step-end infinite' }} />
                                </div>
                            </div>
                        )}

                        {loading && !streamedText && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: '#94a3b8', fontSize: 13 }}>
                                <Loader size={13} style={{ animation: 'spin 1s linear infinite' }} /> Analysing…
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>

                    {/* ── Suggestions ── */}
                    {suggestions.length > 0 && !loading && messages.length < 2 && (
                        <div style={{ padding: '8px 12px', background: '#f1f5f9', borderTop: '1px solid #e2e8f0', display: 'flex', gap: 6, overflowX: 'auto', scrollbarWidth: 'none', flexShrink: 0 }}>
                            {suggestions.map((s, i) => (
                                <button key={i} onClick={() => askGemini(s.text, s.intent)}
                                    style={{ flexShrink: 0, padding: '6px 11px', background: 'white', border: '1px solid #cbd5e1', borderRadius: 14, fontSize: 12, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, color: '#334155', fontWeight: 500, transition: 'all 0.15s', whiteSpace: 'nowrap' }}
                                    onMouseEnter={e => { e.currentTarget.style.background = '#667eea'; e.currentTarget.style.color = 'white'; e.currentTarget.style.borderColor = '#667eea'; }}
                                    onMouseLeave={e => { e.currentTarget.style.background = 'white'; e.currentTarget.style.color = '#334155'; e.currentTarget.style.borderColor = '#cbd5e1'; }}
                                >
                                    {s.icon}{s.text}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* ── Input ── */}
                    <div style={{ padding: '11px 13px', borderTop: '1px solid #e2e8f0', background: 'white', flexShrink: 0 }}>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                            <input
                                ref={inputRef}
                                type="text"
                                placeholder={activeDataset ? `Ask about "${activeDataset.name}"…` : 'Ask about your data…'}
                                value={prompt}
                                onChange={e => setPrompt(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && !loading && askGemini()}
                                disabled={loading}
                                style={{ flex: 1, padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', outline: 'none', fontSize: 13.5, background: loading ? '#f8fafc' : 'white' }}
                            />
                            <button
                                onClick={() => askGemini()}
                                disabled={loading || !prompt.trim()}
                                style={{ padding: '8px 14px', background: loading || !prompt.trim() ? '#cbd5e1' : '#667eea', color: 'white', border: 'none', borderRadius: 8, cursor: loading || !prompt.trim() ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', transition: 'background 0.15s', flexShrink: 0 }}
                            >
                                {loading ? <Loader size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={14} />}
                            </button>
                        </div>
                        {messages.length >= MAX_HISTORY_TURNS * 2 && (
                            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 5, textAlign: 'center' }}>
                                Keeping last {MAX_HISTORY_TURNS} exchanges for efficiency.{' '}
                                <span style={{ cursor: 'pointer', color: '#667eea' }} onClick={() => setMessages([])}>Clear history</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ── FAB ── */}
            <button
                onClick={() => setIsOpen(o => !o)}
                title="Smart Analyst AI"
                style={{ width: 56, height: 56, borderRadius: '50%', background: 'linear-gradient(135deg,#667eea,#764ba2)', color: 'white', border: 'none', boxShadow: '0 4px 14px rgba(102,126,234,0.45)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.25s cubic-bezier(0.4,0,0.2,1)', transform: isOpen ? 'scale(0.88) rotate(90deg)' : 'scale(1)' }}
                onMouseEnter={e => { if (!isOpen) e.currentTarget.style.transform = 'scale(1.08)'; }}
                onMouseLeave={e => { if (!isOpen) e.currentTarget.style.transform = 'scale(1)'; }}
            >
                {isOpen ? <X size={22} /> : <Sparkles size={22} />}
            </button>

            <style>{`
                @keyframes spin  { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
                @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
            `}</style>
        </div>
    );
};