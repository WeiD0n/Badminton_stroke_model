
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
    setPage('Analyse Stroke')
  }

  function resetVideo() {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideo(null)
    setVideoUrl('')
    setMessage('')
    if (fileInput.current) fileInput.current.value = ''
  }

  function startAnalysis() {
    if (!video) {
      setMessage('Upload a badminton video before starting analysis.')
      return
    }

    setMessage(
      'Video selected successfully! ML analysis will be available after the Python backend is connected.'
    )
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
          <section className="page-content inner-page">
            <p className="eyebrow">VIDEO-BASED PERFORMANCE REVIEW</p>
            <h1>Analyse your stroke<span>.</span></h1>
            <p className="subheading">
              Upload a recording to begin your badminton movement analysis.
            </p>

            <section className="upload-panel">
              <div className="panel-heading">
                <div>
                  <h2>Upload your video</h2>
                  <p>Choose a video file from your device.</p>
                </div>
                <span className="muted-tag">GUEST ACCESS</span>
              </div>

              <input
                ref={fileInput}
                type="file"
                accept="video/*"
                hidden
                onChange={(event) => chooseVideo(event.target.files?.[0])}
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
                    <strong>{video.name}</strong>
                    <p>{(video.size / (1024 * 1024)).toFixed(1)} MB</p>
                    <button className="text-button" onClick={resetVideo}>
                      Remove video
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  className="upload-dropzone"
                  onClick={() => fileInput.current?.click()}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault()
                    chooseVideo(event.dataTransfer.files?.[0])
                  }}
                >
                  <span className="upload-icon">↑</span>
                  <strong>Drop your video here</strong>
                  <span>or click to browse your device</span>
                  <span className="upload-formats">VIDEO FILES · MAX 200 MB</span>
                </button>
              )}

              {message && (
                <div className="notice" role="status">{message}</div>
              )}

              <div className="upload-guidelines">
                <h3>For better analysis</h3>
                <ul>
                  <li>Keep the player's full body visible.</li>
                  <li>Use clear footage with minimal obstruction.</li>
                  <li>Record one stroke at a time where possible.</li>
                </ul>
              </div>

              <button
                className="primary-button analyze-button"
                onClick={startAnalysis}
              >
                Analyse video <span>→</span>
              </button>
              <p className="privacy-note">
                Guest results are not saved to an account.
              </p>
            </section>
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
