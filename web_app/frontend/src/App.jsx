
import { useRef, useState } from 'react'
import './App.css'
import badmintonHero from './assets/Neon Badminton Pose Tracking Dashboard.png';

const strokeTypes = [
  { name: 'Smash', icon: '↗', description: 'Powerful attacking stroke' },
  { name: 'Drop Shot', icon: '⌄', description: 'Controlled shot near the net' },
  { name: 'Clear', icon: '↗', description: 'High shot to the backcourt' },
]

function App() {
  const [page, setPage] = useState('Dashboard')
  const [video, setVideo] = useState(null)
  const [videoUrl, setVideoUrl] = useState('')
  const [message, setMessage] = useState('')
  const [selectedStroke, setSelectedStroke] = useState('Smash')
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalysing, setIsAnalysing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const fileInput = useRef(null)

  function chooseVideo(file) {
    if (!file) return

    if (!file.type.startsWith('video/')) {
      setMessage('Please select a valid video file.')
      return
    }

    if (file.size > 200 * 1024 * 1024) {
      setMessage('Please choose a video smaller than 200 MB.')
      return
    }

    if (videoUrl) URL.revokeObjectURL(videoUrl)

    setVideo(file)
    setVideoUrl(URL.createObjectURL(file))
    setMessage('')
    setAnalysisResult(null)
    setPage('Analyse Stroke')
  }

  function resetVideo() {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideo(null)
    setVideoUrl('')
    setMessage('')
    setAnalysisResult(null)
    if (fileInput.current) fileInput.current.value = ''
  }

  async function startAnalysis() {
    if (!video) {
      setMessage('Please upload a video before starting analysis.')
      return
    }

    if (!selectedStroke) {
      setMessage('Please select the stroke type.')
      return
    }

    setIsAnalysing(true)
    setMessage('Connecting to the E05 model. If prompted, select the player in the OpenCV window.')
    setAnalysisResult(null)

    try {
      const formData = new FormData()
      formData.append('file', video)
      formData.append('stroke_type', selectedStroke)

      const response = await fetch('http://127.0.0.1:8000/api/analyse', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || `Analysis failed (HTTP ${response.status}).`)
      }

      setAnalysisResult(data)
      setMessage(data.warning || 'Analysis completed successfully.')
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unknown error.'
      if (detail.includes('Failed to fetch') || detail.includes('NetworkError')) {
        setMessage('Cannot reach the backend. Check that FastAPI is running at http://127.0.0.1:8000 and try again.')
      } else {
        setMessage(`Analysis failed: ${detail}`)
      }
    } finally {
      setIsAnalysing(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setPage('Dashboard')}>
          <span className="brand-icon">R</span>
          <span>
            <strong>RALLY</strong>
            <small>BADMINTON ANALYTICS</small>
          </span>
        </button>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="navigation">
          <button
            className={`nav-item ${page === 'Dashboard' ? 'active' : ''}`}
            onClick={() => setPage('Dashboard')}
          >
            <span>▦</span> Dashboard
          </button>
          <button
            className={`nav-item ${page === 'Analyse Stroke' ? 'active' : ''}`}
            onClick={() => setPage('Analyse Stroke')}
          >
            <span>◎</span> Analyse Stroke
          </button>
          <button
            className={`nav-item ${page === 'History' ? 'active' : ''}`}
            onClick={() => setPage('History')}
          >
            <span>◷</span> Analysis History
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="guest-card">
            <span className="guest-icon">♙</span>
            <strong>Guest mode</strong>
            <p>Analyse strokes without an account.</p>
            <button onClick={() => setPage('Account')}>
              Create free account ↗
            </button>
          </div>
          <div className="sidebar-footer">
            <span className="status-dot" />
            ML SYSTEM <span className="version">V1.0</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <strong>{page}</strong>
          </div>
          <div className="topbar-right">
            <span className="guest-pill">
              <span className="status-dot" /> Guest session
            </span>
            <button
              className="avatar"
              title="Account"
              onClick={() => setPage('Account')}
            >
              G
            </button>
          </div>
        </header>

        {page === 'Dashboard' && (
          <section className="page-content">
            <div className="welcome-row">
              <div>
                <p className="eyebrow">YOUR PERFORMANCE WORKSPACE</p>
                <h1>Welcome to Rally<span>.</span></h1>
                <p className="subheading">
                  Understand your movement. Improve your game.
                </p>
              </div>
              <button
                className="primary-button"
                onClick={() => {
                  setMessage('')
                  setPage('Analyse Stroke')
                }}
              >
                <span>＋</span> Analyse a stroke
              </button>
            </div>

            <section className="hero-card">
              <div className="hero-copy">
                <div className="hero-tag">
                  <span className="status-dot" /> COMPUTER VISION × AI
                </div>
                <h2>Every movement<br />tells a story.</h2>
                <p>
                  Upload your badminton footage to explore stroke
                  classification, body positioning and movement patterns.
                </p>
                <button
                  className="hero-button"
                  onClick={() => setPage('Analyse Stroke')}
                >
                  Start your analysis <span>→</span>
                </button>
                <div className="hero-note">
                  NO ACCOUNT REQUIRED <span>·</span> VIDEO-BASED ANALYSIS
                </div>
              </div>
              <div className="hero-visual">
                <img
                  src={badmintonHero}
                  alt="AI badminton pose tracking illustration"
                  className="badminton-hero-image"
                />
              </div>
            </section>

            <div className="section-heading">
              <div>
                <h2>What you can analyse</h2>
                <p>Explore the three supported badminton stroke categories.</p>
              </div>
              <span className="section-count">03 STROKE TYPES</span>
            </div>

            <div className="stroke-grid">
              {strokeTypes.map((stroke, index) => (
                <button
                  className="stroke-card"
                  key={stroke.name}
                  onClick={() => {
                    setMessage('')
                    setPage('Analyse Stroke')
                  }}
                >
                  <div className={`stroke-icon stroke-${index + 1}`}>
                    {stroke.icon}
                  </div>
                  <div className="stroke-card-text">
                    <span className="stroke-number">0{index + 1}</span>
                    <h3>{stroke.name}</h3>
                    <p>{stroke.description}</p>
                  </div>
                  <span className="card-arrow">↗</span>
                </button>
              ))}
            </div>

            <div className="bottom-grid">
              <section className="panel">
                <div className="panel-heading">
                  <h2>Analysis workflow</h2>
                  <span className="muted-tag">HOW IT WORKS</span>
                </div>
                <div className="workflow">
                  {[
                    ['01', 'Upload video', 'Select recorded footage'],
                    ['02', 'Extract pose', 'Track skeletal landmarks'],
                    ['03', 'Evaluate stroke', 'Review model predictions'],
                  ].map(([number, title, description]) => (
                    <div className="workflow-step" key={number}>
                      <span className="step-number">{number}</span>
                      <div>
                        <strong>{title}</strong>
                        <p>{description}</p>
                      </div>
                      <span className="step-arrow">→</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="panel account-panel">
                <div className="account-symbol">↗</div>
                <p className="eyebrow">YOUR TRAINING JOURNEY</p>
                <h2>Build consistency,<br />one stroke at a time.</h2>
                <p className="account-description">
                  Create an account to save your analyses and review
                  your progress over time.
                </p>
                <button
                  className="secondary-button"
                  onClick={() => setPage('Account')}
                >
                  Explore account options →
                </button>
              </section>
            </div>

            <footer className="page-footer">
              <span>RALLY <span className="footer-dot">●</span> BADMINTON STROKE EVALUATION</span>
              <span>RESEARCH PROJECT · V1.0</span>
            </footer>
          </section>
        )}

        {page === 'Analyse Stroke' && (
          <section className="page-content inner-page upload-page">
            <div className="upload-page-heading">
              <p className="eyebrow">VIDEO-BASED PERFORMANCE REVIEW</p>
              <h1>Analyse your stroke<span>.</span></h1>
              <p className="subheading">
                Upload your footage, select your stroke, and prepare for analysis.
              </p>
            </div>

            <div className="upload-layout">
              <section className="upload-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Upload your video</h2>
                    <p>Choose a recording from your device.</p>
                  </div>
                  <span className="muted-tag">GUEST ACCESS</span>
                </div>

                <input
                  ref={fileInput}
                  type="file"
                  accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,video/*"
                  hidden
                  onChange={(event) => {
                    chooseVideo(event.target.files?.[0])
                    event.target.value = ''
                  }}
                />

                {video ? (
                  <div className="selected-video">
                    <video
                      className="video-preview"
                      src={videoUrl}
                      controls
                      playsInline
                    />

                    <div className="selected-video-info">
                      <div className="file-details">
                        <span className="file-check">✓</span>
                        <div>
                          <strong title={video.name}>{video.name}</strong>
                          <p>
                            {(video.size / (1024 * 1024)).toFixed(1)} MB · Video ready
                          </p>
                        </div>
                      </div>

                      <div className="video-actions">
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => fileInput.current?.click()}
                        >
                          Replace video
                        </button>
                        <button
                          type="button"
                          className="text-button"
                          onClick={resetVideo}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className={`upload-dropzone ${isDragging ? 'dragging' : ''}`}
                    onClick={() => fileInput.current?.click()}
                    onDragEnter={(event) => {
                      event.preventDefault()
                      setIsDragging(true)
                    }}
                    onDragOver={(event) => event.preventDefault()}
                    onDragLeave={(event) => {
                      if (!event.currentTarget.contains(event.relatedTarget)) {
                        setIsDragging(false)
                      }
                    }}
                    onDrop={(event) => {
                      event.preventDefault()
                      setIsDragging(false)
                      chooseVideo(event.dataTransfer.files?.[0])
                    }}
                  >
                    <span className="upload-icon">
                      <svg
                        viewBox="0 0 24 24"
                        width="30"
                        height="30"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.7"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <path d="M12 16V4" />
                        <path d="m7 9 5-5 5 5" />
                        <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
                      </svg>
                    </span>
                    <strong>
                      {isDragging ? 'Drop your video here' : 'Drag and drop your video'}
                    </strong>
                    <span className="upload-description">
                      or <span className="browse-link">browse files</span> on your device
                    </span>
                    <span className="upload-formats">
                      MP4 · MOV · WEBM · AVI
                    </span>
                    <span className="upload-limit">Maximum file size: 200 MB</span>
                  </button>
                )}

                <div className="stroke-selection">
                  <div className="section-heading">
                    <div>
                      <h2>Select stroke type</h2>
                      <p>Choose the stroke you want to review.</p>
                    </div>
                    <span className="muted-tag">REQUIRED</span>
                  </div>

                  <div className="stroke-choice-grid">
                    {strokeTypes.map((stroke, index) => (
                      <button
                        type="button"
                        key={stroke.name}
                        className={`stroke-choice ${
                          selectedStroke === stroke.name ? 'selected' : ''
                        }`}
                        aria-pressed={selectedStroke === stroke.name}
                        onClick={() => {
                          setSelectedStroke(stroke.name)
                          setMessage('')
                          setAnalysisResult(null)
                        }}
                      >
                        <span className={`stroke-icon stroke-${index + 1}`}>
                          {stroke.icon}
                        </span>
                        <span className="stroke-choice-text">
                          <strong>{stroke.name}</strong>
                          <small>{stroke.description}</small>
                        </span>
                        <span className="selection-indicator">
                          {selectedStroke === stroke.name ? '✓' : ''}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {message && (
                  <div className="notice" role="status" aria-live="polite">
                    <span>ⓘ</span>
                    {message}
                  </div>
                )}

                {analysisResult && (
                  <section className="panel analysis-results" aria-live="polite">
                    <div className="panel-heading">
                      <div>
                        <p className="eyebrow">E05 MODEL RESULTS</p>
                        <h2>Stroke analysis</h2>
                      </div>
                      <span className="muted-tag">COMPLETED</span>
                    </div>
                    <div className="analysis-prediction">
                      <span className="result-label">Predicted stroke</span>
                      <h3>{analysisResult.predicted_stroke}</h3>
                      <p>{Math.round((analysisResult.confidence || 0) * 100)}% model confidence</p>
                    </div>
                    <div className="probability-list">
                      {Object.entries(analysisResult.probabilities || {}).map(([name, value]) => (
                        <div className="probability-row" key={name}>
                          <div className="probability-heading">
                            <span>{name}</span>
                            <strong>{Math.round(value * 100)}%</strong>
                          </div>
                          <div className="probability-track">
                            <div className="probability-fill" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="result-meta">
                      <span>Pose detection</span>
                      <strong>{Math.round((analysisResult.video_info?.pose_detection_rate || 0) * 100)}%</strong>
                      <span>Frames processed</span>
                      <strong>{analysisResult.video_info?.frame_count ?? '—'}</strong>
                      <span>Tracker failures</span>
                      <strong>{analysisResult.video_info?.tracker_failures ?? '—'}</strong>
                    </div>
                    {analysisResult.matches_selected_type === false && (
                      <p className="result-disclaimer">The predicted stroke differs from your selected stroke type. Check the clip and player tracking.</p>
                    )}
                    <p className="result-disclaimer">Confidence is the model's predicted probability, not a guarantee of real-world accuracy.</p>
                  </section>
                )}

                <div className="upload-guidelines">
                  <h3>
                    <span></span> Recording tips for better results
                  </h3>
                  <ul>
                    <li>Keep the player's full body visible throughout the clip.</li>
                    <li>Use clear footage with good lighting and minimal obstruction.</li>
                    <li>Keep the camera steady and record one stroke at a time where possible.</li>
                  </ul>
                </div>

                <button
                  type="button"
                  className="primary-button analyze-button"
                  onClick={startAnalysis}
                  disabled={isAnalysing}
                >
                  {isAnalysing ? 'Analysing video…' : 'Analyse video'} <span>{isAnalysing ? '⏳' : '→'}</span>
                </button>

                <p className="privacy-note">
                  Guest mode · Your selected video is not saved to an account.
                </p>
              </section>

              <aside className="upload-side-panel">
                <div className="upload-side-icon">◎</div>
                <p className="eyebrow">YOUR ANALYSIS</p>
                <h2>From footage to feedback.</h2>
                <p>
                  Prepare your badminton video for pose tracking and stroke
                  classification.
                </p>

                <div className="upload-progress-step">
                  <span>01</span>
                  <div>
                    <strong>Upload footage</strong>
                    <small>
                      {video ? 'Video selected' : 'Waiting for your video'}
                    </small>
                  </div>
                  {video && <span className="step-complete">✓</span>}
                </div>

                <div className="upload-progress-step">
                  <span>02</span>
                  <div>
                    <strong>Choose stroke</strong>
                    <small>{selectedStroke} selected</small>
                  </div>
                  <span className="step-complete">✓</span>
                </div>

                <div className="upload-progress-step">
                  <span>03</span>
                  <div>
                    <strong>Evaluate movement</strong>
                    <small>{isAnalysing ? 'Analysis in progress' : analysisResult ? 'Analysis completed' : 'Ready when you start analysis'}</small>
                  </div>
                </div>

                <div className="guest-info">
                  <strong>Guest session</strong>
                  <p>You can prepare a video without creating an account.</p>
                </div>
              </aside>
            </div>
          </section>
        )}


        {page === 'History' && (
          <section className="page-content inner-page">
            <p className="eyebrow">YOUR ACTIVITY</p>
            <h1>Analysis history<span>.</span></h1>
            <section className="empty-state panel">
              <span className="empty-icon">◷</span>
              <h2>No saved analyses yet</h2>
              <p>
                Guest analyses are temporary. Create an account to save
                future results and review them here.
              </p>
              <button className="primary-button" onClick={() => setPage('Analyse Stroke')}>
                Analyse a stroke →
              </button>
            </section>
          </section>
        )}

        {page === 'Account' && (
          <section className="page-content inner-page">
            <p className="eyebrow">YOUR ACCOUNT</p>
            <h1>Keep your progress<span>.</span></h1>
            <section className="account-options panel">
              <div className="account-symbol">♙</div>
              <h2>Continue as a guest</h2>
              <p>
                You can analyse videos without registering. Guest results
                are not saved to an account.
              </p>
              <button className="primary-button" onClick={() => setPage('Analyse Stroke')}>
                Continue analysing →
              </button>
              <div className="account-divider"><span>OR</span></div>
              <h2>Save your analyses</h2>
              <p>
                Account registration and sign-in will be connected to the
                backend in a later step.
              </p>
              <button className="secondary-button" onClick={() => setMessage('Account registration is not connected yet.')}>
                Register / Log in
              </button>
              {message && <div className="notice" role="status">{message}</div>}
            </section>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
