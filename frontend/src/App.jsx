import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import {
  Activity, BarChart3, Bell, CheckCircle2, ChevronDown, Database,
  Download, FileUp, Gauge, Leaf, Menu, RefreshCw, Search, Settings2,
  Sprout, Target, TrendingDown, TrendingUp, UploadCloud, X
} from 'lucide-react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis, BarChart, Bar
} from 'recharts'

const API = 'http://localhost:5000/api'
const money = (n) => Number(n).toLocaleString('en-IN', { maximumFractionDigits: 2 })
const strategyLabels = {
  target_only: 'Target-only',
  correlation: 'Correlation',
  granger: 'Granger',
  hybrid: 'Hybrid',
  all_markets: 'All markets'
}
const strategyDescriptions = {
  target_only: 'Uses only Pratapgarh historical price and calendar features.',
  correlation: 'Selects external markets using strongest absolute lagged correlations.',
  granger: 'Selects markets using the lowest Granger-test p-values.',
  hybrid: 'Combines percentile ranks of available correlation and Granger scores.',
  all_markets: 'Uses all eligible external markets as feature sources.'
}

function MetricCard({ icon: Icon, label, value, note, tone = 'green' }) {
  return <div className="metric-card">
    <div className={`metric-icon ${tone}`}><Icon size={19} /></div>
    <div className="metric-label">{label}</div>
    <div className="metric-value">{value}</div>
    <div className="metric-note">{note}</div>
  </div>
}

function ResultsTable({ rows }) {
  if (!rows?.length) return <div className="empty-small">Run the experiment to view results.</div>
  return <div className="table-wrap"><table>
    <thead><tr><th>Strategy</th><th>Model</th><th>External markets</th><th>MAE ↓</th><th>RMSE ↓</th><th>MAPE ↓</th><th>R² ↑</th></tr></thead>
    <tbody>{rows.map((r, i) => <tr key={`${r.strategy}-${r.model}`}>
      <td><span className={i === 0 ? 'rank first' : 'rank'}>{i + 1}</span>{strategyLabels[r.strategy] || r.strategy}</td>
      <td>{r.model}</td><td>{r.external_markets}</td>
      <td>{money(r.MAE)}</td><td className={i === 0 ? 'best-number' : ''}>{money(r.RMSE)}</td>
      <td>{Number(r.MAPE).toFixed(2)}%</td><td>{Number(r.R2).toFixed(4)}</td>
    </tr>)}</tbody>
  </table></div>
}

export default function App() {
  const [file, setFile] = useState(null)
  const [metadata, setMetadata] = useState(null)
  const [commodity, setCommodity] = useState('Onion')
  const [market, setMarket] = useState('Pratapgarh')
  const [marketSearch, setMarketSearch] = useState('')
  const [topK, setTopK] = useState(5)
  const [minOverlap, setMinOverlap] = useState(60)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('overview')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const markets = useMemo(() => {
    if (!metadata || !commodity) return []
    // The API provides commodity counts; market names are loaded through a lightweight
    // metadata extension below when a CSV is uploaded.
    return metadata.markets || []
  }, [metadata, commodity])

  const filteredMarkets = useMemo(() => {
    const q = marketSearch.toLowerCase()
    return markets.filter(m => m.toLowerCase().includes(q)).slice(0, 100)
  }, [markets, marketSearch])

  useEffect(() => {
    axios.get(`${API}/metadata`).then(({ data }) => {
      setMetadata(data)
      if (data.commodities?.includes('Onion')) setCommodity('Onion')
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!metadata) return
    axios.get(`${API}/markets`, { params: { commodity } }).then(({ data }) => {
      setMetadata(prev => ({ ...prev, markets: data.markets }))
      if (data.markets?.length && !data.markets.includes(market)) {
        setMarket(data.markets.includes('Pratapgarh') ? 'Pratapgarh' : data.markets[0])
      }
    }).catch(() => setMetadata(prev => ({ ...prev, markets: [] })))
  }, [commodity])

  const uploadDataset = async () => {
    if (!file) return
    setUploading(true); setError(''); setResult(null)
    try {
      const body = new FormData()
      body.append('file', file)
      const { data } = await axios.post(`${API}/upload`, body)
      setMetadata(data)
      const defaultCommodity = data.commodities.includes('Onion') ? 'Onion' : data.commodities[0]
      setCommodity(defaultCommodity)
      setMarket(data.markets?.includes('Pratapgarh') ? 'Pratapgarh' : (data.markets?.[0] || ''))
    } catch (e) {
      setError(e.response?.data?.error || 'Dataset upload failed.')
    } finally { setUploading(false) }
  }

  const runExperiment = async () => {
    setLoading(true); setError(''); setResult(null)
    try {
      const { data } = await axios.post(`${API}/run`, {
        target_market: market, commodity, top_k: topK, min_overlap: minOverlap
      })
      setResult(data)
      setActiveTab('overview')
    } catch (e) {
      setError(e.response?.data?.error || 'Experiment failed.')
    } finally { setLoading(false) }
  }

  const exportResults = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'mandi-forecast-results.json'; a.click()
    URL.revokeObjectURL(url)
  }

  const predictionChart = result?.selected_model_test_predictions?.dates?.map((date, i) => ({
    date: date.slice(5), actual: result.selected_model_test_predictions.actual[i],
    predicted: result.selected_model_test_predictions.predicted[i]
  })) || []

  const bestTest = result?.test_results?.[0]
  const selectedTest = result?.test_results?.find(r =>
    r.strategy === result.selected_on_validation.strategy &&
    r.model === result.selected_on_validation.model
  )

  return <div className="app-shell">
    <aside className={`sidebar ${sidebarOpen ? '' : 'closed'}`}>
      <div className="brand"><div className="brand-mark"><Sprout size={23}/></div>
        {sidebarOpen && <div><b>Mandi<span>Forecast</span></b><small>AGRI INTELLIGENCE</small></div>}
      </div>
      <div className="side-caption">WORKSPACE</div>
      <button className={`side-link ${activeTab === 'overview' ? 'selected' : ''}`} onClick={() => setActiveTab('overview')}><BarChart3 size={18}/>{sidebarOpen && 'Overview'}</button>
      <button className={`side-link ${activeTab === 'markets' ? 'selected' : ''}`} onClick={() => setActiveTab('markets')}><Database size={18}/>{sidebarOpen && 'Market selection'}</button>
      <button className={`side-link ${activeTab === 'evaluation' ? 'selected' : ''}`} onClick={() => setActiveTab('evaluation')}><Gauge size={18}/>{sidebarOpen && 'Model evaluation'}</button>
      <div className="sidebar-bottom">
        <div className="side-caption">PROJECT</div>
        <div className="project-chip"><Leaf size={17}/>{sidebarOpen && <span>Onion · Pratapgarh</span>}</div>
        {sidebarOpen && <p className="side-footnote">Cross-market selection for agricultural mandi price forecasting.</p>}
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div className="top-left">
          <button className="icon-btn" onClick={() => setSidebarOpen(v => !v)}><Menu size={19}/></button>
          <div><div className="eyebrow">RESEARCH WORKSPACE</div><h2>Cross-Market Forecasting</h2></div>
        </div>
        <div className="top-actions">
          <span className="status-dot"></span><span className="backend-label">Local analysis</span>
          <button className="icon-btn" title="Refresh metadata" onClick={() => axios.get(`${API}/metadata`).then(({data}) => setMetadata(data)).catch(() => {})}><RefreshCw size={17}/></button>
          <div className="user-avatar">MF</div>
        </div>
      </header>

      <div className="content">
        <section className="page-intro">
          <div><div className="eyebrow green-text">AGRICULTURAL PRICE INTELLIGENCE</div>
            <h1>Mandi price forecasting</h1>
            <p>Evaluate whether selected external mandis improve onion price forecasts at a target market.</p>
          </div>
          <div className="intro-badge"><Activity size={16}/> Research experiment</div>
        </section>

        <section className="upload-panel">
          <div className="upload-icon"><UploadCloud size={23}/></div>
          <div className="upload-copy">
            <b>{metadata ? 'Dataset loaded' : 'Upload agricultural price dataset'}</b>
            <span>{metadata ? `${money(metadata.rows)} cleaned records · ${metadata.commodities?.length || 0} commodities · ${metadata.date_min} to ${metadata.date_max}` : 'Choose Agriculture_price_dataset.csv to run the original notebook workflow.'}</span>
          </div>
          <label className="file-button"><FileUp size={16}/> Choose CSV<input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)}/></label>
          <button className="primary-button upload-submit" disabled={!file || uploading} onClick={uploadDataset}>{uploading ? 'Uploading…' : 'Load dataset'}</button>
        </section>

        {error && <div className="error-box"><X size={17}/>{error}</div>}

        <section className="config-panel">
          <div className="section-heading"><div><h3>Experiment configuration</h3><p>Uses the same chronological split and selection logic as the notebook.</p></div><span className="config-tag">Notebook-aligned</span></div>
          <div className="config-grid">
            <div className="field"><label>Commodity</label>
              <select value={commodity} onChange={e => {setCommodity(e.target.value); setResult(null);}}>
                {(metadata?.commodities || ['Onion']).map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div className="field"><label>Target mandi</label>
              <div className="search-select"><Search size={15}/><input list="market-options" value={market} onChange={e => {setMarket(e.target.value);setResult(null);}} placeholder="Search or type a mandi…"/>
                <datalist id="market-options">
                  {markets.map(m => <option key={m} value={m} />)}
                </datalist>
              </div>
              <small>{markets.length ? `${markets.length.toLocaleString()} markets available for this commodity` : 'Upload CSV to load all market names.'}</small>
            </div>
            <div className="field"><label>Top-K external mandis</label><input type="number" min="1" max="20" value={topK} onChange={e => setTopK(Number(e.target.value))}/></div>
            <div className="field"><label>Minimum overlap</label><input type="number" min="20" value={minOverlap} onChange={e => setMinOverlap(Number(e.target.value))}/></div>
          </div>
          <div className="config-footer">
            <div className="split-note"><span className="split-dot"></span> 60% train <span className="split-dot amber"></span> 20% validation <span className="split-dot blue"></span> 20% test</div>
            <button className="primary-button run-button" disabled={!metadata || loading} onClick={runExperiment}>
              {loading ? <><span className="spinner"></span> Running models…</> : <><TrendingUp size={17}/> Run experiment</>}
            </button>
          </div>
          {!metadata && <div className="helper-note">Upload the dataset first. Default research setup is Pratapgarh + Onion, Top-K 5, minimum overlap 60.</div>}
        </section>

        {result && <>
          <section className="result-summary">
            <div className="summary-heading"><div><div className="eyebrow green-text">EXPERIMENT COMPLETE</div><h3>{result.config.target_market} · {result.config.commodity}</h3></div>
              <button className="secondary-button" onClick={exportResults}><Download size={16}/> Export JSON</button></div>
            <div className="summary-grid">
              <MetricCard icon={Target} label="Target observations" value={money(result.config.target_observations)} note="Reported target prices" />
              <MetricCard icon={Database} label="Eligible external mandis" value={money(result.config.eligible_markets)} note="Passed minimum overlap" tone="blue"/>
              <MetricCard icon={CheckCircle2} label="Validation winner" value={strategyLabels[result.selected_on_validation.strategy]} note={`${result.selected_on_validation.model} · RMSE ${money(result.selected_on_validation.validation_rmse)}`} tone="orange"/>
              <MetricCard icon={TrendingUp} label="Best test RMSE" value={money(bestTest?.RMSE)} note={`${strategyLabels[bestTest?.strategy]} · ${bestTest?.model}`} tone="purple"/>
            </div>
          </section>

          <section className="tab-row">
            <button className={activeTab === 'overview' ? 'active' : ''} onClick={() => setActiveTab('overview')}>Overview</button>
            <button className={activeTab === 'markets' ? 'active' : ''} onClick={() => setActiveTab('markets')}>Market selection</button>
            <button className={activeTab === 'evaluation' ? 'active' : ''} onClick={() => setActiveTab('evaluation')}>Model evaluation</button>
          </section>

          {activeTab === 'overview' && <>
            <div className="two-col">
              <div className="panel">
                <div className="panel-title"><div><h3>Held-out test forecast</h3><p>Validation-selected model · actual vs predicted</p></div><span className="tiny-tag">TEST · 20%</span></div>
                {predictionChart.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={predictionChart}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="date" tick={{fontSize:10}}/><YAxis tick={{fontSize:10}}/><Tooltip/><Legend/><Line type="monotone" dataKey="actual" name="Actual price" stroke="#254f3e" strokeWidth={2.5} dot={false}/><Line type="monotone" dataKey="predicted" name="Predicted price" stroke="#e5a343" strokeWidth={2.5} strokeDasharray="5 4" dot={false}/></LineChart></ResponsiveContainer></div> : <div className="empty-small">No prediction series returned for the validation-selected model.</div>}
                {selectedTest && <div className="selected-test-note">Validation-selected combination test RMSE: <b>{money(selectedTest.RMSE)}</b> · MAE: <b>{money(selectedTest.MAE)}</b> · R²: <b>{Number(selectedTest.R2).toFixed(4)}</b></div>}
              </div>
              <div className="panel">
                <div className="panel-title"><div><h3>Target price regimes</h3><p>Observed price distribution by chronological split</p></div></div>
                <div className="regime-list">{Object.entries(result.price_distribution).map(([name, s]) => <div className="regime" key={name}>
                  <div className="regime-top"><b>{name}</b><span>{s.count} observations</span></div>
                  <div className="regime-value">Mean <strong>₹{money(s.mean)}</strong><span>Median ₹{money(s.median)}</span></div>
                  <div className="regime-bar"><span style={{width:`${Math.max(4, Math.min(100, s.mean / Math.max(...Object.values(result.price_distribution).map(x=>x.mean)) * 100))}%`}}/></div>
                </div>)}</div>
                <div className="method-note">Different price distributions across time can affect validation-based model selection. The chronological split is retained.</div>
              </div>
            </div>
            <div className="panel table-panel"><div className="panel-title"><div><h3>Held-out test comparison</h3><p>All strategy/model combinations, sorted by test RMSE</p></div></div><ResultsTable rows={result.test_results}/></div>
          </>}

          {activeTab === 'markets' && <div className="two-col">
            {Object.entries(result.selection).map(([key, list]) => <div className="panel selection-panel" key={key}>
              <div className="panel-title"><div><h3>{strategyLabels[key] || key}</h3><p>{strategyDescriptions[key]}</p></div><span className="count-pill">{list.length} markets</span></div>
              {list.length ? <div className="market-list">{list.slice(0, 30).map((m, i) => <div className="market-row" key={m.market}><span className="market-rank">{String(i+1).padStart(2,'0')}</span><b>{m.market}</b>{i === 0 && key !== 'target_only' && <span className="top-label">Top ranked</span>}</div>)}
                {list.length > 30 && <small className="muted">Showing first 30 of {list.length} eligible markets.</small>}</div> : <div className="empty-small">No external markets used.</div>}
            </div>)}
            <div className="panel"><div className="panel-title"><div><h3>Top correlation candidates</h3><p>Absolute lagged correlation scores calculated on training data</p></div></div>
              {result.top_correlation.map(x => <div className="score-row" key={x.market}><span>{x.market}</span><b>{x.score.toFixed(4)}</b></div>)}
            </div>
            <div className="panel"><div className="panel-title"><div><h3>Top Granger candidates</h3><p>Minimum p-value across tested lags</p></div></div>
              {result.top_granger.length ? result.top_granger.map(x => <div className="score-row" key={x.market}><span>{x.market}</span><b>{x.p_value.toExponential(3)}</b></div>) : <div className="empty-small">Granger testing unavailable or no valid scores.</div>}
            </div>
          </div>}

          {activeTab === 'evaluation' && <>
            <div className="panel table-panel"><div className="panel-title"><div><h3>Validation results</h3><p>Used to select the preferred strategy/model by lowest validation RMSE</p></div><span className="tiny-tag">MODEL SELECTION</span></div><ResultsTable rows={result.validation_results}/></div>
            <div className="panel table-panel"><div className="panel-title"><div><h3>Held-out test results</h3><p>Final chronological test comparison; not used to select the model</p></div><span className="tiny-tag">FINAL EVALUATION</span></div><ResultsTable rows={result.test_results}/></div>
            <div className="panel"><div className="panel-title"><div><h3>Price-distribution diagnostic</h3><p>Descriptive statistics for the three time periods</p></div></div><div className="table-wrap"><table><thead><tr><th>Split</th><th>Count</th><th>Mean</th><th>Median</th><th>Std. dev.</th><th>Min</th><th>Max</th></tr></thead><tbody>{Object.entries(result.price_distribution).map(([name,s])=><tr key={name}><td>{name}</td><td>{s.count}</td><td>{money(s.mean)}</td><td>{money(s.median)}</td><td>{money(s.std)}</td><td>{money(s.min)}</td><td>{money(s.max)}</td></tr>)}</tbody></table></div></div>
          </>}
        </>}

        {!result && <section className="welcome-state">
          <div className="welcome-art"><Sprout size={42}/></div>
          <h3>Your mandi research workspace</h3>
          <p>Upload the original CSV, choose a target mandi and commodity, then run the five market-selection strategies with Random Forest and XGBoost.</p>
          <div className="workflow"><span><b>01</b> Upload dataset</span><span><b>02</b> Configure target</span><span><b>03</b> Run & compare</span></div>
        </section>}

        <footer className="footer"><span>Cross-Market Selection for Agricultural Mandi Price Forecasting</span><span>Research prototype · Chronological evaluation</span></footer>
      </div>
    </main>
  </div>
}
