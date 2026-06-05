import React, { useState, useEffect, useRef } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  AreaChart, Area
} from 'recharts';
import { 
  TrendingUp, Users, Activity, ShieldAlert, Cpu, 
  Play, RefreshCw, Send, Radio
} from 'lucide-react';

const API_BASE = "";        // proxied by Vite → http://localhost:8000
const WS_BASE  = "";        // ws:// proxied by Vite

// Interface Definitions
interface PredictionRecord {
  id: number;
  user_id: string;
  predicted_probability: number;
  fatigue_flag: number;
  risk_level: string;
  business_archetype: string;
  timestamp: string;
  retention_action_triggered: string | null;
  conversion_success: number | null;
}

export default function App() {
  // Authentication State
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [username, setUsername] = useState('strategyx_admin');
  const [password, setPassword] = useState('strategyx_password');
  const [authError, setAuthError] = useState('');

  // Active Tab
  const [activeTab, setActiveTab] = useState<'dashboard' | 'simulator' | 'streaming'>('dashboard');

  // Dashboard Metrics & Logs
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [dbStats, _setDbStats] = useState({
    totalUsers: 15000,
    fatiguedCount: 3120,
    riskRate: 20.8,
    avgScore: 32.4
  });

  // What-If Simulator Settings
  const [simUserId, setSimUserId] = useState('U_MANUAL_REACT');
  const [simTenure, setSimTenure] = useState(120);
  const [simTier, setSimTier] = useState('Standard');
  const [simMin30, setSimMin30] = useState(30.0);
  const [simMin7, setSimMin7] = useState(12.0);
  const [simSess30, _setSimSess30] = useState(12);
  const [simSess7, _setSimSess7] = useState(2);
  const [simCompletion, setSimCompletion] = useState(0.25);
  const [simGenres, _setSimGenres] = useState(3);
  const [simDaysSince, setSimDaysSince] = useState(6);
  const [simBinges, _setSimBinges] = useState(1);
  const [simPeak, _setSimPeak] = useState(70.0);
  const [simOriginals, _setSimOriginals] = useState(45.0);
  const [simClickRate, setSimClickRate] = useState(0.12);

  // Simulator Prediction Output
  const [simOutput, setSimOutput] = useState<any>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [feedbackSuccess, setFeedbackSuccess] = useState<number | null>(null);
  const [feedbackAction, setFeedbackAction] = useState('Discount Voucher');

  // WebSocket Live Streaming Feed
  const [streamingEvents, setStreamingEvents] = useState<any[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Handle Login
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError('');
    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const res = await fetch(`${API_BASE}/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        localStorage.setItem('token', data.access_token);
        setToken(data.access_token);
      } else {
        setAuthError('Incorrect username or password');
      }
    } catch (err) {
      setAuthError('Failed to connect to authentication server.');
    }
  };

  // Fetch Database Logs History
  const fetchLogs = async () => {
    if (!token) return;
    setLoadingHistory(true);
    try {
      // Mock stats for SQLite sandbox display since REST API logs represent user sessions
      // We can also fetch actual stats from Swagger
      const mockHistory: PredictionRecord[] = [
        { id: 1, user_id: "U938210", predicted_probability: 0.82, fatigue_flag: 1, risk_level: "High Risk", business_archetype: "Binge-and-Leave", timestamp: "2026-06-05 23:55:12", retention_action_triggered: "Auto-queue similar binging content", conversion_success: 1 },
        { id: 2, user_id: "U123490", predicted_probability: 0.12, fatigue_flag: 0, risk_level: "Low Risk", business_archetype: "Active & Engaged", timestamp: "2026-06-05 23:42:01", retention_action_triggered: null, conversion_success: null },
        { id: 3, user_id: "U872516", predicted_probability: 0.54, fatigue_flag: 1, risk_level: "Medium Risk", business_archetype: "Frustrated Browser", timestamp: "2026-06-05 23:30:19", retention_action_triggered: "Discount popup & genre recommendations", conversion_success: 0 },
        { id: 4, user_id: "U661209", predicted_probability: 0.61, fatigue_flag: 1, risk_level: "High Risk", business_archetype: "Waning Casual", timestamp: "2026-06-05 23:15:44", retention_action_triggered: "Monitor only", conversion_success: null },
      ];
      setHistory(mockHistory);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingHistory(false);
    }
  };

  // Run Simulator Prediction
  const runSimulator = async () => {
    if (!token) return;
    setSimLoading(true);
    setFeedbackSuccess(null);
    try {
      const payload = {
        user_id: simUserId,
        tenure_days: simTenure,
        subscription_tier: simTier,
        avg_daily_minutes_last_7d: simMin7,
        avg_daily_minutes_last_30d: simMin30,
        sessions_last_7d: simSess7,
        sessions_last_30d: simSess30,
        avg_completion_rate: simCompletion,
        unique_genres_watched_30d: simGenres,
        days_since_last_session: simDaysSince,
        binge_sessions_last_30d: simBinges,
        peak_hour_viewing_pct: simPeak,
        original_content_pct: simOriginals,
        recommendation_click_rate: simClickRate
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: json_payload_formatter(payload)
      });

      if (res.ok) {
        const data = await res.json();
        // Fetch SHAP drivers
        const explainRes = await fetch(`${API_BASE}/explain`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: json_payload_formatter(payload)
        });
        const explainData = await explainRes.json();
        
        setSimOutput({
          ...data,
          shaps: explainData.slice(0, 5) // Top 5
        });
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSimLoading(false);
    }
  };

  // Helper JSON formatter
  const json_payload_formatter = (obj: any) => {
    return JSON.stringify(obj);
  };

  // Submit Feedback Conversion Action
  const submitFeedback = async (success: number) => {
    if (!token || !simOutput) return;
    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          user_id: simOutput.user_id,
          conversion_success: success,
          retention_action_triggered: feedbackAction
        })
      });
      if (res.ok) {
        setFeedbackSuccess(success);
        fetchLogs();
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Setup WebSocket connection
  const toggleWebSocket = () => {
    if (wsConnected) {
      if (wsRef.current) wsRef.current.close();
      setWsConnected(false);
      return;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/predict`);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      console.log("WebSocket connected.");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStreamingEvents(prev => [data, ...prev.slice(0, 19)]); // Store last 20
    };

    ws.onclose = () => {
      setWsConnected(false);
      console.log("WebSocket disconnected.");
    };
  };

  // Push mock event over WebSocket
  const sendMockWsEvent = () => {
    if (!wsRef.current || !wsConnected) return;
    const randomUser = `U${Math.floor(Math.random() * 900000) + 100000}`;
    const payload = {
      user_id: randomUser,
      tenure_days: Math.floor(Math.random() * 400) + 10,
      subscription_tier: ["Basic", "Standard", "Premium"][Math.floor(Math.random() * 3)],
      avg_daily_minutes_last_7d: parseFloat((Math.random() * 45).toFixed(1)),
      avg_daily_minutes_last_30d: parseFloat((Math.random() * 50).toFixed(1)),
      sessions_last_7d: Math.floor(Math.random() * 10),
      sessions_last_30d: Math.floor(Math.random() * 40) + 10,
      avg_completion_rate: parseFloat(Math.random().toFixed(2)),
      unique_genres_watched_30d: Math.floor(Math.random() * 8) + 1,
      days_since_last_session: Math.floor(Math.random() * 10),
      binge_sessions_last_30d: Math.floor(Math.random() * 5),
      peak_hour_viewing_pct: parseFloat((Math.random() * 100).toFixed(1)),
      original_content_pct: parseFloat((Math.random() * 100).toFixed(1)),
      recommendation_click_rate: parseFloat(Math.random().toFixed(2))
    };
    wsRef.current.send(JSON.stringify(payload));
  };

  useEffect(() => {
    if (token) {
      fetchLogs();
    }
  }, [token]);

  // Handle Logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
  };

  // Login Screen
  if (!token) {
    return (
      <div className="min-h-screen bg-darkBg text-white flex items-center justify-center font-sans">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.2),rgba(255,255,255,0))]" />
        <div className="w-full max-w-md bg-darkCard border border-gray-800 rounded-2xl p-8 shadow-2xl relative z-10">
          <div className="flex justify-center mb-6">
            <div className="w-12 h-12 bg-gradient-to-tr from-accentBlue to-accentPurple rounded-xl flex items-center justify-center shadow-lg">
              <Cpu className="w-6 h-6 text-white" />
            </div>
          </div>
          <h2 className="text-3xl font-extrabold text-center bg-clip-text text-transparent bg-gradient-to-r from-accentBlue to-accentPurple mb-2">
            StrategyX
          </h2>
          <p className="text-gray-400 text-center text-sm mb-8">
            OTT subscriber fatigue classification control center
          </p>

          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label className="block text-xs uppercase tracking-wider text-gray-400 mb-2 font-bold">Username</label>
              <input 
                type="text" 
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-black border border-gray-800 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-accentBlue transition-colors"
                required
              />
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-gray-400 mb-2 font-bold">Password</label>
              <input 
                type="password" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-black border border-gray-800 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-accentBlue transition-colors"
                required
              />
            </div>
            {authError && (
              <div className="text-red-500 text-xs font-bold text-center bg-red-900/20 border border-red-900/50 py-2 rounded-lg">
                {authError}
              </div>
            )}
            <button 
              type="submit"
              className="w-full bg-gradient-to-r from-accentBlue to-accentPurple hover:opacity-90 transition-opacity text-white font-bold py-3 px-4 rounded-lg shadow-lg"
            >
              Sign In
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Dashboard Metrics Data
  const segmentData = [
    { name: 'Active & Engaged', count: 4800, color: '#10b981' },
    { name: 'Waning Casual', count: 3500, color: '#f59e0b' },
    { name: 'Frustrated Browser', count: 2800, color: '#8b5cf6' },
    { name: 'Binge-and-Leave', count: 3900, color: '#ef4444' },
  ];

  return (
    <div className="min-h-screen bg-darkBg text-white font-sans relative">
      {/* Top Navbar */}
      <nav className="border-b border-gray-800 bg-black/40 backdrop-blur-md sticky top-0 z-50 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-tr from-accentBlue to-accentPurple rounded-lg flex items-center justify-center">
            <Cpu className="w-4 h-4 text-white" />
          </div>
          <span className="font-extrabold text-xl bg-clip-text text-transparent bg-gradient-to-r from-accentBlue to-accentPurple">
            StrategyX Control Panel
          </span>
        </div>

        <div className="flex items-center space-x-6">
          <button 
            onClick={() => setActiveTab('dashboard')}
            className={`text-sm font-bold transition-colors ${activeTab === 'dashboard' ? 'text-accentBlue' : 'text-gray-400 hover:text-white'}`}
          >
            Dashboard
          </button>
          <button 
            onClick={() => setActiveTab('simulator')}
            className={`text-sm font-bold transition-colors ${activeTab === 'simulator' ? 'text-accentBlue' : 'text-gray-400 hover:text-white'}`}
          >
            What-If Simulator
          </button>
          <button 
            onClick={() => setActiveTab('streaming')}
            className={`text-sm font-bold transition-colors ${activeTab === 'streaming' ? 'text-accentBlue' : 'text-gray-400 hover:text-white'}`}
          >
            WS Live Feed
          </button>
          <button 
            onClick={handleLogout}
            className="border border-gray-800 hover:border-red-900 hover:text-red-400 transition-colors rounded-lg px-4 py-2 text-xs font-bold"
          >
            Sign Out
          </button>
        </div>
      </nav>

      {/* Main Content Body */}
      <main className="max-w-7xl mx-auto px-8 py-10 relative z-10">
        
        {/* EXECUTIVE DASHBOARD */}
        {activeTab === 'dashboard' && (
          <div className="space-y-8 animate-fadeIn">
            {/* Page Header */}
            <div>
              <h1 className="text-4xl font-extrabold text-white">System Executive Dashboard</h1>
              <p className="text-gray-400 text-sm mt-1">Multi-container database log aggregates, model metrics, and active retention action statistics.</p>
            </div>

            {/* KPI Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-darkCard border border-gray-800 rounded-xl p-6 flex items-center space-x-4 hover:border-accentBlue transition-colors">
                <div className="p-3 bg-accentBlue/10 text-accentBlue rounded-lg">
                  <Users className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-gray-400 text-xs uppercase font-bold">Subscribers base</p>
                  <h3 className="text-2xl font-bold">{dbStats.totalUsers.toLocaleString()}</h3>
                </div>
              </div>

              <div className="bg-darkCard border border-gray-800 rounded-xl p-6 flex items-center space-x-4 hover:border-accentBlue transition-colors">
                <div className="p-3 bg-red-500/10 text-red-500 rounded-lg">
                  <ShieldAlert className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-gray-400 text-xs uppercase font-bold">At-Risk Fatigued</p>
                  <h3 className="text-2xl font-bold">{dbStats.fatiguedCount.toLocaleString()}</h3>
                </div>
              </div>

              <div className="bg-darkCard border border-gray-800 rounded-xl p-6 flex items-center space-x-4 hover:border-accentBlue transition-colors">
                <div className="p-3 bg-accentPurple/10 text-accentPurple rounded-lg">
                  <TrendingUp className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-gray-400 text-xs uppercase font-bold">Risk Proportion</p>
                  <h3 className="text-2xl font-bold">{dbStats.riskRate}%</h3>
                </div>
              </div>

              <div className="bg-darkCard border border-gray-800 rounded-xl p-6 flex items-center space-x-4 hover:border-accentBlue transition-colors">
                <div className="p-3 bg-yellow-500/10 text-yellow-500 rounded-lg">
                  <Activity className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-gray-400 text-xs uppercase font-bold">Average fatigue score</p>
                  <h3 className="text-2xl font-bold">{dbStats.avgScore}%</h3>
                </div>
              </div>
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-darkCard border border-gray-800 rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4">Subscriber Archetypes Breakdown</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={segmentData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                      <XAxis type="number" stroke="#9ca3af" />
                      <YAxis dataKey="name" type="category" stroke="#9ca3af" width={130} />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#333' }} />
                      <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                        {segmentData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-darkCard border border-gray-800 rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4">Fatigue Density Distribution</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={[
                        { density: 0.1, count: 50 },
                        { density: 0.2, count: 200 },
                        { density: 0.3, count: 480 },
                        { density: 0.4, count: 650 },
                        { density: 0.5, count: 800 },
                        { density: 0.6, count: 500 },
                        { density: 0.7, count: 310 },
                        { density: 0.8, count: 180 },
                        { density: 0.9, count: 90 },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                      <XAxis dataKey="density" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#333' }} />
                      <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="rgba(59, 130, 246, 0.2)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Inferences & Feedback Logs Table */}
            <div className="bg-darkCard border border-gray-800 rounded-xl p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold">Prediction Conversion & Intervention Logs</h3>
                <button 
                  onClick={fetchLogs}
                  className="p-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg flex items-center space-x-2 text-xs font-bold transition-colors"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loadingHistory ? 'animate-spin' : ''}`} />
                  <span>Refresh Logs</span>
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-gray-300">
                  <thead className="text-xs uppercase bg-black/60 text-gray-400">
                    <tr>
                      <th className="px-6 py-4">Subscriber ID</th>
                      <th className="px-6 py-4">Probability</th>
                      <th className="px-6 py-4">Risk Category</th>
                      <th className="px-6 py-4">Archetype</th>
                      <th className="px-6 py-4">Intervention Triggered</th>
                      <th className="px-6 py-4">A/B Conversion</th>
                      <th className="px-6 py-4">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {history.map((row) => (
                      <tr key={row.id} className="hover:bg-gray-900/30">
                        <td className="px-6 py-4 font-mono font-bold text-white">{row.user_id}</td>
                        <td className="px-6 py-4">{(row.predicted_probability * 100).toFixed(0)}%</td>
                        <td className="px-6 py-4">
                          <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                            row.risk_level === 'High Risk' ? 'bg-red-900/35 text-red-400 border border-red-800/40' :
                            row.risk_level === 'Medium Risk' ? 'bg-yellow-900/35 text-yellow-400 border border-yellow-800/40' :
                            'bg-green-900/35 text-green-400 border border-green-800/40'
                          }`}>
                            {row.risk_level}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-accentPurple font-bold">{row.business_archetype}</td>
                        <td className="px-6 py-4 text-xs max-w-xs truncate">{row.retention_action_triggered || 'None'}</td>
                        <td className="px-6 py-4">
                          {row.conversion_success === 1 && (
                            <span className="text-green-400 font-bold bg-green-950/40 px-2 py-1 border border-green-800/30 rounded">Converted</span>
                          )}
                          {row.conversion_success === 0 && (
                            <span className="text-red-400 font-bold bg-red-950/40 px-2 py-1 border border-red-800/30 rounded">Churned</span>
                          )}
                          {row.conversion_success === null && (
                            <span className="text-gray-500">Untracked</span>
                          )}
                        </td>
                        <td className="px-6 py-4 text-xs font-mono text-gray-500">{row.timestamp}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* CHURN WHAT-IF SIMULATOR */}
        {activeTab === 'simulator' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 animate-fadeIn">
            {/* Input Sliders */}
            <div className="lg:col-span-2 bg-darkCard border border-gray-800 rounded-xl p-6 space-y-6">
              <h3 className="text-lg font-bold">Simulator Telemetry Settings</h3>
              
              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Subscriber ID</label>
                <input 
                  type="text" 
                  value={simUserId} 
                  onChange={(e) => setSimUserId(e.target.value)}
                  className="w-full bg-black border border-gray-800 rounded px-3 py-2 text-sm text-white"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Tenure (Days): {simTenure}</label>
                <input 
                  type="range" min="1" max="1000" value={simTenure} 
                  onChange={(e) => setSimTenure(parseInt(e.target.value))}
                  className="w-full accent-accentBlue"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Subscription Tier</label>
                <select 
                  value={simTier} 
                  onChange={(e) => setSimTier(e.target.value)}
                  className="w-full bg-black border border-gray-800 rounded px-3 py-2 text-sm text-white"
                >
                  <option value="Basic">Basic</option>
                  <option value="Standard">Standard</option>
                  <option value="Premium">Premium</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Completion Rate: {(simCompletion * 100).toFixed(0)}%</label>
                <input 
                  type="range" min="0" max="100" value={simCompletion * 100} 
                  onChange={(e) => setSimCompletion(parseFloat(e.target.value) / 100)}
                  className="w-full accent-accentBlue"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Avg Daily Minutes (30d): {simMin30}</label>
                <input 
                  type="range" min="1" max="180" value={simMin30} 
                  onChange={(e) => setSimMin30(parseInt(e.target.value))}
                  className="w-full accent-accentBlue"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Avg Daily Minutes (7d): {simMin7}</label>
                <input 
                  type="range" min="0" max="180" value={simMin7} 
                  onChange={(e) => setSimMin7(parseInt(e.target.value))}
                  className="w-full accent-accentBlue"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Rec Click Rate: {(simClickRate * 100).toFixed(0)}%</label>
                <input 
                  type="range" min="0" max="100" value={simClickRate * 100} 
                  onChange={(e) => setSimClickRate(parseFloat(e.target.value) / 100)}
                  className="w-full accent-accentBlue"
                />
              </div>

              <div>
                <label className="block text-xs text-gray-400 uppercase font-bold mb-2">Days Since Last Session: {simDaysSince}</label>
                <input 
                  type="range" min="0" max="30" value={simDaysSince} 
                  onChange={(e) => setSimDaysSince(parseInt(e.target.value))}
                  className="w-full accent-accentBlue"
                />
              </div>

              <button 
                onClick={runSimulator}
                disabled={simLoading}
                className="w-full bg-accentBlue hover:bg-accentBlue/90 transition-colors text-white font-bold py-2.5 rounded flex items-center justify-center space-x-2"
              >
                <Cpu className="w-4 h-4" />
                <span>{simLoading ? 'Analyzing...' : 'Calculate Churn Risk'}</span>
              </button>
            </div>

            {/* Results & Actions Panel */}
            <div className="lg:col-span-3 space-y-6">
              {!simOutput ? (
                <div className="bg-darkCard border border-gray-800 rounded-xl p-12 text-center text-gray-500">
                  Adjust sliders and trigger prediction calculation to check fatigue risk output.
                </div>
              ) : (
                <div className="space-y-6 animate-fadeIn">
                  {/* Gauge Probability Panel */}
                  <div className="bg-darkCard border border-gray-800 rounded-xl p-6">
                    <h3 className="text-lg font-bold mb-4">Diagnostics Output</h3>
                    
                    <div className="space-y-3">
                      <div className="flex justify-between items-center text-sm">
                        <span>Risk Probability</span>
                        <span className="font-mono text-xl font-bold">{(simOutput.fatigue_probability * 100).toFixed(1)}%</span>
                      </div>
                      {/* Bar Gauge */}
                      <div className="w-full h-4 bg-black/60 rounded-full overflow-hidden border border-gray-800 relative">
                        <div 
                          className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500" 
                          style={{ width: `${simOutput.fatigue_probability * 100}%` }}
                        />
                        <div 
                          className="absolute top-0 bottom-0 w-0.5 bg-white" 
                          style={{ left: `${simOutput.optimal_threshold || 45}%` }}
                          title="Model Threshold Boundary"
                        />
                      </div>
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>Low Risk</span>
                        <span>Threshold Boundary</span>
                        <span>High Churn Risk</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mt-6">
                      <div className="bg-black/40 border border-gray-850 p-4 rounded-lg">
                        <span className="text-gray-400 text-xs">Risk Level</span>
                        <p className="font-extrabold text-lg mt-1 text-white">{simOutput.risk_level}</p>
                      </div>
                      <div className="bg-black/40 border border-gray-850 p-4 rounded-lg">
                        <span className="text-gray-400 text-xs">Archetype Classification</span>
                        <p className="font-extrabold text-lg mt-1 text-accentPurple">{simOutput.business_archetype}</p>
                      </div>
                    </div>
                  </div>

                  {/* SHAP Explainer */}
                  <div className="bg-darkCard border border-gray-800 rounded-xl p-6">
                    <h3 className="text-lg font-bold mb-4">SHAP Explainable AI Feature Drivers</h3>
                    <div className="space-y-4">
                      {simOutput.shaps.map((item: any, idx: number) => {
                        const positive = item.shap_value > 0;
                        return (
                          <div key={idx} className="space-y-1">
                            <div className="flex justify-between text-xs font-mono">
                              <span className="capitalize text-gray-400">{item.feature.replace(/_/g, ' ')}</span>
                              <span className={positive ? 'text-red-400' : 'text-green-400'}>
                                {positive ? '+' : ''}{item.shap_value.toFixed(3)}
                              </span>
                            </div>
                            <div className="w-full bg-black/40 h-2 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${positive ? 'bg-red-500' : 'bg-green-500'}`}
                                style={{ 
                                  width: `${Math.min(Math.abs(item.shap_value) * 100, 100)}%`,
                                  marginLeft: positive ? '0' : 'auto' 
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Targeted Retention Action triggering and Conversion Feedback */}
                  <div className="bg-darkCard border border-gray-800 rounded-xl p-6">
                    <h3 className="text-lg font-bold mb-4 font-outfit tracking-tight">Targeted Retention Action (A/B Test Blueprint)</h3>
                    
                    <div className="space-y-4">
                      <div>
                        <label className="block text-xs uppercase font-bold text-gray-400 mb-2">Intervention Blueprint Action</label>
                        <select 
                          value={feedbackAction}
                          onChange={(e) => setFeedbackAction(e.target.value)}
                          className="w-full bg-black border border-gray-850 text-sm py-2 px-3 rounded text-white"
                        >
                          <option value="Discount Voucher">Discount Voucher (50% Off next month)</option>
                          <option value="Push Notifications Recommendations">Push Notifications: Recommendations for Watchlist</option>
                          <option value="Weekly Original Content Releases">Flagship Originals: Weekly Release cadence</option>
                          <option value="Monitor Only">Monitor Only (Control Group)</option>
                        </select>
                      </div>

                      <div className="bg-black/30 p-4 border border-gray-850 rounded-lg">
                        <span className="text-xs font-bold text-accentPurple">Action Blueprint Details:</span>
                        <p className="text-gray-400 text-xs mt-1">
                          {feedbackAction === 'Discount Voucher' && 'Mitigates monetary churn index. Highly effective for Basic subscribers.'}
                          {feedbackAction === 'Push Notifications Recommendations' && 'Re-engages Frustrated Browsers by highlighting content fitting their genre mix.'}
                          {feedbackAction === 'Weekly Original Content Releases' && 'Binds Binge-and-Leave subscribers to longer subscription durations.'}
                          {feedbackAction === 'Monitor Only' && 'Acts as baseline control group subscriber tracking.'}
                        </p>
                      </div>

                      <div className="border-t border-gray-800 pt-6">
                        <span className="block text-xs uppercase font-bold text-gray-400 mb-3 text-center">Log A/B Conversion Feedback</span>
                        
                        <div className="flex justify-center space-x-4">
                          <button 
                            onClick={() => submitFeedback(1)}
                            className="bg-green-600 hover:bg-green-700 transition-colors text-white font-bold py-2 px-6 rounded text-sm flex items-center space-x-2"
                          >
                            <Play className="w-4 h-4" />
                            <span>Subscriber Converted (Stayed)</span>
                          </button>
                          <button 
                            onClick={() => submitFeedback(0)}
                            className="bg-red-600 hover:bg-red-700 transition-colors text-white font-bold py-2 px-6 rounded text-sm flex items-center space-x-2"
                          >
                            <ShieldAlert className="w-4 h-4" />
                            <span>Subscriber Churned (Left)</span>
                          </button>
                        </div>
                        
                        {feedbackSuccess !== null && (
                          <div className="mt-4 text-center text-xs font-bold text-green-400 animate-fadeIn">
                            Feedback updated successfully: {feedbackSuccess === 1 ? 'Converted' : 'Churned'}. Retraining metrics refreshed!
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* WEBSOCKET LIVE STREAM FEED */}
        {activeTab === 'streaming' && (
          <div className="space-y-8 animate-fadeIn">
            {/* Page Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-4xl font-extrabold text-white">Live WebSocket Telemetry Feed</h1>
                <p className="text-gray-400 text-sm mt-1">Consumes simulated streaming events from Redpanda and maps model fatigue probabilities in real-time.</p>
              </div>
              
              <div className="flex space-x-4">
                <button 
                  onClick={toggleWebSocket}
                  className={`px-5 py-2.5 rounded-lg flex items-center space-x-2 text-sm font-bold shadow-lg transition-colors ${
                    wsConnected ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <Radio className={`w-4 h-4 ${wsConnected ? 'animate-pulse' : ''}`} />
                  <span>{wsConnected ? 'Disconnect WS' : 'Connect Live Feed'}</span>
                </button>
                
                {wsConnected && (
                  <button 
                    onClick={sendMockWsEvent}
                    className="bg-accentBlue hover:bg-accentBlue/90 px-5 py-2.5 rounded-lg text-sm font-bold flex items-center space-x-2 transition-colors"
                  >
                    <Send className="w-4 h-4" />
                    <span>Send Mock Event</span>
                  </button>
                )}
              </div>
            </div>

            {/* Connection status banner */}
            <div className={`p-4 rounded-xl border text-sm font-bold flex items-center justify-between ${
              wsConnected ? 'bg-green-950/20 border-green-900/50 text-green-400' : 'bg-red-950/20 border-red-900/50 text-red-400'
            }`}>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500 animate-ping' : 'bg-red-500'}`} />
                <span>Status: {wsConnected ? 'ACTIVE' : 'OFFLINE'}</span>
              </div>
              <span>{wsConnected ? `Streaming on ${WS_BASE}/ws/predict` : 'Connect to listen to event stream'}</span>
            </div>

            {/* Streaming log feed */}
            <div className="bg-darkCard border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-bold mb-4">Live Classification Events</h3>
              
              {streamingEvents.length === 0 ? (
                <div className="py-20 text-center text-gray-500">
                  {wsConnected ? 'Connected. Click "Send Mock Event" or run streaming producer script.' : 'Offline. Click "Connect Live Feed".'}
                </div>
              ) : (
                <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                  {streamingEvents.map((ev, index) => (
                    <div key={index} className="bg-black/40 border border-gray-850 p-4 rounded-lg flex items-center justify-between hover:border-gray-800 transition-colors animate-slideIn">
                      <div className="flex items-center space-x-6">
                        <span className="font-mono text-sm font-bold text-accentBlue">{ev.user_id}</span>
                        <div className="text-xs">
                          <span className="text-gray-500 block">Archetype</span>
                          <span className="font-bold text-accentPurple">{ev.business_archetype}</span>
                        </div>
                        <div className="text-xs">
                          <span className="text-gray-500 block">Risk Category</span>
                          <span className={`font-bold ${ev.risk_level === 'High Risk' ? 'text-red-400' : 'text-green-400'}`}>{ev.risk_level}</span>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <span className="font-mono font-bold text-sm bg-gray-900 border border-gray-850 px-3 py-1 rounded">
                          {(ev.fatigue_probability * 100).toFixed(0)}% Score
                        </span>
                        
                        {ev.is_fatigued ? (
                          <span className="bg-red-900/30 text-red-400 border border-red-800/50 px-2 py-0.5 rounded text-xxs font-bold uppercase tracking-wider">Action Triggered</span>
                        ) : (
                          <span className="bg-green-900/30 text-green-400 border border-green-800/50 px-2 py-0.5 rounded text-xxs font-bold uppercase tracking-wider">Active & Safe</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
