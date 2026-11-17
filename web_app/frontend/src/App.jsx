
import { useEffect, useRef, useState } from 'react'
import './App.css'
import badmintonHero from './assets/Neon Badminton Pose Tracking Dashboard.png';

const strokeTypes = [
  { name: 'Smash', icon: '↗', description: 'Powerful attacking stroke' },
  { name: 'Drop Shot', icon: '↗', description: 'Controlled shot near the net' },
  { name: 'Clear', icon: '↗', description: 'High shot to the backcourt' },
]

function App() {
  // Restore the last visited page after a browser refresh in the same tab.
  const [page, setPage] = useState(() => {
    const validPages = [
      'Dashboard',
      'Analyse Stroke',
      'History',
      'Compare Analyses',
      'Account',
    ]

    try {
      const savedPage = sessionStorage.getItem('rally_active_page')
      return validPages.includes(savedPage) ? savedPage : 'Dashboard'
    } catch {
      return 'Dashboard'
    }
  })
  const [video, setVideo] = useState(null)
  const [videoUrl, setVideoUrl] = useState('')
  const [message, setMessage] = useState('')
  const [selectedStroke, setSelectedStroke] = useState('Smash')
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalysing, setIsAnalysing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [showTechnicalBreakdown, setShowTechnicalBreakdown] = useState(false)
  const [skeletonOverlayUrl, setSkeletonOverlayUrl] = useState('')
  const [isSkeletonOverlayLoading, setIsSkeletonOverlayLoading] = useState(false)
  const [isReportLoading, setIsReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  const [skeletonOverlayError, setSkeletonOverlayError] = useState('')
  const [skeletonOverlayInfo, setSkeletonOverlayInfo] = useState(null)
  const [selectedRoi, setSelectedRoi] = useState(null)
  const [isDrawingRoi, setIsDrawingRoi] = useState(false)
  const roiStartRef = useRef(null)
  const roiStageRef = useRef(null)
  const [saveAnalysis, setSaveAnalysis] = useState(false)
  const [saveVideo, setSaveVideo] = useState(false)
  const [historyItems, setHistoryItems] = useState([])
  const [isHistoryLoading, setIsHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [selectedHistory, setSelectedHistory] = useState(null)
  const [historyVideo, setHistoryVideo] = useState(null)
  const [isHistoryVideoLoading, setIsHistoryVideoLoading] = useState(false)
  const [historyVideoError, setHistoryVideoError] = useState('')
  // Keep selected comparison sessions when the user refreshes the page.
  const [comparisonIds, setComparisonIds] = useState(() => {
    try {
      const savedIds = JSON.parse(sessionStorage.getItem('rally_comparison_ids') || '[]')
      return Array.isArray(savedIds) ? savedIds.slice(0, 5) : []
    } catch {
      return []
    }
  })
  const [comparisonStrokeFilter, setComparisonStrokeFilter] = useState('')
  const [comparisonError, setComparisonError] = useState('')
  const [dashboardLoading, setDashboardLoading] = useState(false)
  const [dashboardError, setDashboardError] = useState('')
  const fileInput = useRef(null)

  const API_URL = import.meta.env.VITE_API_URL || 'https://rally-badminton-api-151529283119.asia-southeast1.run.app';

  const [user, setUser] = useState(null)
  const [isSessionLoading, setIsSessionLoading] = useState(true)

  const [authMode, setAuthMode] = useState('login')
  const [authEmail, setAuthEmail] = useState('')
  const [authUsername, setAuthUsername] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authMessage, setAuthMessage] = useState('')
  const [isAuthLoading, setIsAuthLoading] = useState(false)

  // Persist the active page so refreshing History, Analyse Stroke,
  // Compare Analyses, or Account keeps the user on that page.
  useEffect(() => {
    try {
      sessionStorage.setItem('rally_active_page', page)
    } catch (error) {
      console.warn('Could not preserve the current page:', error)
    }
  }, [page])

  useEffect(() => {
    try {
      sessionStorage.setItem('rally_comparison_ids', JSON.stringify(comparisonIds))
    } catch (error) {
      console.warn('Could not preserve selected comparison sessions:', error)
    }
  }, [comparisonIds])

  const validFormAnalyses = historyItems
    .filter((item) => item.form_score !== null && item.form_score !== undefined && Number.isFinite(Number(item.form_score)))
    .slice()
    .sort((a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0))
  const formScores = validFormAnalyses.map((item) => Number(item.form_score))
  const averageFormScore = formScores.length
    ? formScores.reduce((total, score) => total + score, 0) / formScores.length
    : null
  const bestFormScore = formScores.length ? Math.max(...formScores) : null
  const averageScore = (field) => {
    const values = historyItems
      .filter((item) => item[field] !== null && item[field] !== undefined && item[field] !== '')
      .map((item) => Number(item[field]))
      .filter((value) => Number.isFinite(value))
    return values.length ? values.reduce((total, value) => total + value, 0) / values.length : null
  }
  const strokeProgress = ['Smash', 'Drop Shot', 'Clear'].map((stroke) => {
    const scores = historyItems
      .filter((item) => (item.stroke_type || '').toLowerCase() === stroke.toLowerCase())
      .filter((item) => item.form_score !== null && item.form_score !== undefined && item.form_score !== '')
      .map((item) => Number(item.form_score))
      .filter((score) => Number.isFinite(score))
    return {
      name: stroke,
      count: historyItems.filter((item) => (item.stroke_type || '').toLowerCase() === stroke.toLowerCase()).length,
      average: scores.length ? scores.reduce((total, score) => total + score, 0) / scores.length : null,
    }
  })
  const chartPoints = validFormAnalyses.map((item, index) => {
    const x = validFormAnalyses.length === 1 ? 300 : 36 + (index * 528) / (validFormAnalyses.length - 1)
    const y = 174 - (Math.max(0, Math.min(100, Number(item.form_score))) / 100) * 140
    return { x, y, score: Number(item.form_score), date: item.created_at }
  })
  const selectedComparisons = historyItems
    .filter((item) => comparisonIds.includes(item.id))
    .slice()
    .sort((a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0))
  const comparisonCandidates = historyItems.filter((item) =>
    item.form_score !== null && item.form_score !== undefined && Number.isFinite(Number(item.form_score))
  )
  const comparisonStrokeOptions = [...new Set(comparisonCandidates.map((item) => item.stroke_type).filter(Boolean))]
  const effectiveComparisonStroke = comparisonStrokeFilter || comparisonStrokeOptions.find((stroke) =>
    comparisonCandidates.filter((item) => (item.stroke_type || '').toLowerCase() === stroke.toLowerCase()).length >= 2
  ) || comparisonStrokeOptions[0] || ''
  const filteredComparisonCandidates = comparisonCandidates.filter((item) =>
    (item.stroke_type || '').toLowerCase() === effectiveComparisonStroke.toLowerCase()
  )
  const comparisonStroke = selectedComparisons[0]?.stroke_type || effectiveComparisonStroke

  // Restore the authenticated session from the HTTP-only cookie on page load.
  useEffect(() => {
    let isMounted = true

    async function restoreSession() {
      try {
        const response = await fetch(`${API_URL}/api/auth/me`, {
          credentials: 'include',
        })
        if (!response.ok) {
          if (response.status !== 401) {
            throw new Error(`Session check failed (HTTP ${response.status}).`)
          }
          return
        }
        const profile = await response.json()
        if (isMounted) setUser(profile)
      } catch (error) {
        console.error('Could not restore login session:', error)
      } finally {
        if (isMounted) setIsSessionLoading(false)
      }
    }

    restoreSession()
    return () => { isMounted = false }
  }, [])

  // Release generated overlay URLs when replaced or when the component unmounts.
  useEffect(() => {
    return () => {
      if (skeletonOverlayUrl) URL.revokeObjectURL(skeletonOverlayUrl)
    }
  }, [skeletonOverlayUrl])

  // Load this account's saved evaluations for the progress dashboard.
  // The existing history endpoint enforces ownership on the backend.
  useEffect(() => {
    let isMounted = true

    async function loadDashboardProgress() {
      if (!user) {
        setHistoryItems([])
        setDashboardError('')
        setDashboardLoading(false)
        return
      }

      setDashboardLoading(true)
      setDashboardError('')
      try {
        const response = await fetch(`${API_URL}/api/analyses?limit=100`, {
          credentials: 'include',
        })
        const data = await response.json().catch(() => ({}))
        if (!response.ok) {
          throw new Error(data.detail || `Could not load progress (HTTP ${response.status}).`)
        }
        if (isMounted) {
          setHistoryItems(Array.isArray(data.items) ? data.items : [])
        }
      } catch (error) {
        if (isMounted) {
          setDashboardError(error.message || 'Could not load your progress data.')
        }
      } finally {
        if (isMounted) setDashboardLoading(false)
      }
    }

    loadDashboardProgress()
    return () => { isMounted = false }
  }, [user, API_URL])

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
    setSkeletonOverlayUrl('')
    setSkeletonOverlayError('')
    setSkeletonOverlayInfo(null)
    setSelectedRoi(null)
    setPage('Analyse Stroke')
  }

  function resetVideo() {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideo(null)
    setVideoUrl('')
    setMessage('')
    setAnalysisResult(null)
    setSkeletonOverlayUrl('')
    setSkeletonOverlayError('')
    setSkeletonOverlayInfo(null)
    setSelectedRoi(null)
    setIsDrawingRoi(false)
    if (fileInput.current) fileInput.current.value = ''
  }
  
  async function handleAuthSubmit(event) {
    event.preventDefault()
    setAuthMessage('')
    setIsAuthLoading(true)

    try {
      if (authMode === 'register') {
        const registerResponse = await fetch(
          `${API_URL}/api/auth/register`,
          {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              email: authEmail,
              username: authUsername,
              password: authPassword,
            }),
          }
        )

        const registerData = await registerResponse.json()

        if (!registerResponse.ok) {
          throw new Error(
            registerData.detail || 'Registration failed.'
          )
        }
      }

      // Log in after registration, or directly for existing users.
      const loginResponse = await fetch(
        `${API_URL}/api/auth/login`,
        {
          method: 'POST',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: authEmail,
            password: authPassword,
          }),
        }
      )

      const loginData = await loginResponse.json()

      if (!loginResponse.ok) {
        throw new Error(loginData.detail || 'Login failed.')
      }

      // Retrieve the authenticated user's profile.
      const profileResponse = await fetch(
        `${API_URL}/api/auth/me`,
        {
          credentials: 'include',
        }
      )

      const profileData = await profileResponse.json()

      if (!profileResponse.ok) {
        throw new Error(
          profileData.detail || 'Could not retrieve your account.'
        )
      }

      setUser(profileData)

      setAuthPassword('')
      setAuthMessage(
        authMode === 'register'
          ? 'Account created successfully!'
          : 'Welcome back!'
      )

    } catch (error) {
      setAuthMessage(error.message || 'Something went wrong.')
    } finally {
      setIsAuthLoading(false)
    }
  }

  async function handleLogout() {
    try {
      await fetch(`${API_URL}/api/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      })
    } catch (error) {
      console.error('Logout request failed:', error)
    }
    setUser(null)
    setAuthEmail('')
    setAuthUsername('')
    setAuthPassword('')
    setAuthMessage('')
    setSaveAnalysis(false)
    setSaveVideo(false)
    setHistoryItems([])
    setComparisonIds([])
    setComparisonStrokeFilter('')
    setComparisonError('')
    try {
      sessionStorage.removeItem('rally_comparison_ids')
      sessionStorage.removeItem('rally_active_page')
    } catch (error) {
      console.warn('Could not clear saved comparison state:', error)
    }
    setSelectedHistory(null)
    setHistoryVideo(null)
    setHistoryVideoError('')
    setPage('Dashboard')
  }


  async function loadHistory() {
    if (!user) {
      setHistoryItems([])
      setHistoryError('Please log in to view your saved analyses.')
      setPage('History')
      return
    }

    setPage('History')
    setIsHistoryLoading(true)
    setHistoryError('')
    setSelectedHistory(null)

    try {
      const response = await fetch(`${API_URL}/api/analyses`, {
        credentials: 'include',
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || `Could not load history (HTTP ${response.status}).`)
      }
      setHistoryItems(Array.isArray(data.items) ? data.items : [])
    } catch (error) {
      setHistoryError(error.message || 'Could not load analysis history.')
    } finally {
      setIsHistoryLoading(false)
    }
  }

  async function loadComparisonHistory() {
    if (!user) {
      setPage('Compare Analyses')
      setComparisonError('Please log in to compare your saved analyses.')
      setHistoryItems([])
      return
    }

    setPage('Compare Analyses')
    setIsHistoryLoading(true)
    setHistoryError('')
    setComparisonError('')
    setComparisonIds([])
    setComparisonStrokeFilter('')

    try {
      const response = await fetch(`${API_URL}/api/analyses?limit=100`, {
        credentials: 'include',
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || `Could not load analyses (HTTP ${response.status}).`)
      }
      setHistoryItems(Array.isArray(data.items) ? data.items : [])
    } catch (error) {
      setComparisonError(error.message || 'Could not load your saved analyses.')
    } finally {
      setIsHistoryLoading(false)
    }
  }

  function toggleComparisonSelection(item) {
    setComparisonError('')
    if (comparisonIds.includes(item.id)) {
      setComparisonIds(comparisonIds.filter((id) => id !== item.id))
      return
    }
    if (comparisonIds.length >= 5) {
      setComparisonError('You can compare up to five sessions at a time.')
      return
    }
    const firstSelected = historyItems.find((entry) => entry.id === comparisonIds[0])
    if (firstSelected && (firstSelected.stroke_type || '').toLowerCase() !== (item.stroke_type || '').toLowerCase()) {
      setComparisonError(`Please select sessions with the same stroke type (${firstSelected.stroke_type}).`)
      return
    }
    setComparisonIds([...comparisonIds, item.id])
  }

  async function openHistoryItem(item) {
    if (!user) return
    setHistoryError('')
    try {
      const response = await fetch(`${API_URL}/api/analyses/${item.id}`, {
        credentials: 'include',
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load this analysis.')
      }
      // Reuse the same results interface shown after a new analysis.
      // The history endpoint wraps the original analysis payload in `result`.
      const savedResult = data.result ?? data
      setAnalysisResult(savedResult)
      setSelectedHistory(null)
      setMessage('Loaded your saved analysis from history.')
      setPage('Analyse Stroke')
    } catch (error) {
      setHistoryError(error.message || 'Could not load this analysis.')
    }
  }

  async function openHistoryVideo(item) {
    if (!user || !item.video_saved) return

    setIsHistoryVideoLoading(true)
    setHistoryVideoError('')
    setHistoryVideo(null)

    try {
      const response = await fetch(
        `${API_URL}/api/analyses/${item.id}/video`,
        { credentials: 'include' }
      )
      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        throw new Error(data.detail || 'Could not retrieve this video.')
      }

      setHistoryVideo({
        url: data.video_url,
        filename: data.filename || item.video_filename || 'Badminton analysis video',
        analysisId: item.id,
      })
    } catch (error) {
      setHistoryVideoError(error.message || 'Could not retrieve this video.')
    } finally {
      setIsHistoryVideoLoading(false)
    }
  }

  // ROI coordinates are stored as normalised values (0 to 1), independent
  // of the displayed video size. The backend converts them to source pixels.
  function getRoiPoint(event) {
    const bounds = roiStageRef.current?.getBoundingClientRect()
    if (!bounds || !bounds.width || !bounds.height) return null

    return {
      x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
    }
  }

  function handleRoiPointerDown(event) {
    if (!video || isAnalysing) return
    const point = getRoiPoint(event)
    if (!point) return

    event.preventDefault()
    event.currentTarget.setPointerCapture?.(event.pointerId)
    roiStartRef.current = point
    setSelectedRoi({ x: point.x, y: point.y, width: 0, height: 0 })
    setIsDrawingRoi(true)
    setMessage('Drag around the player’s full visible body, then release to set the selection.')
    setAnalysisResult(null)
  }

  function handleRoiPointerMove(event) {
    if (!isDrawingRoi || !roiStartRef.current) return
    const point = getRoiPoint(event)
    if (!point) return

    const start = roiStartRef.current
    setSelectedRoi({
      x: Math.min(start.x, point.x),
      y: Math.min(start.y, point.y),
      width: Math.abs(point.x - start.x),
      height: Math.abs(point.y - start.y),
    })
  }

  function handleRoiPointerUp(event) {
    if (!isDrawingRoi || !roiStartRef.current) return
    handleRoiPointerMove(event)
    setIsDrawingRoi(false)
    roiStartRef.current = null
  }

  function clearRoi() {
    setSelectedRoi(null)
    setMessage('Draw a selection box around the player before starting analysis.')
  }


  async function downloadAnalysisReport(analysisData = null, analysisId = null) {
    setIsReportLoading(true)
    setReportError('')

    try {
      const payload = analysisId
        ? { analysis_id: analysisId }
        : { analysis_data: analysisData }

      const response = await fetch(`${API_URL}/api/analyses/report`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Could not generate the analysis report.')
      }

      const blob = await response.blob()
      if (!blob.size || (blob.type && !blob.type.includes('pdf'))) {
        throw new Error('The server did not return a valid PDF report.')
      }

      const objectUrl = URL.createObjectURL(blob)
      const link = document.createElement('a')
      const stroke = analysisData?.selected_stroke_type || selectedHistory?.stroke_type || 'analysis'
      const safeStroke = String(stroke).toLowerCase().replace(/[^a-z0-9_-]+/g, '_')
      link.href = objectUrl
      link.download = `rally_${safeStroke || 'analysis'}_report.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(objectUrl)
    } catch (error) {
      setReportError(error.message || 'Could not download the analysis report.')
    } finally {
      setIsReportLoading(false)
    }
  }

  async function generateSkeletonOverlay() {
    if (!video) {
      setSkeletonOverlayError('The original uploaded video is no longer available. Please upload it again.')
      return
    }

    if (!selectedRoi) {
      setSkeletonOverlayError('The player selection is missing. Please select the player again.')
      return
    }

    setIsSkeletonOverlayLoading(true)
    setSkeletonOverlayError('')
    setSkeletonOverlayInfo(null)

    try {
      const formData = new FormData()
      formData.append('file', video)
      formData.append('roi_x', String(selectedRoi.x))
      formData.append('roi_y', String(selectedRoi.y))
      formData.append('roi_width', String(selectedRoi.width))
      formData.append('roi_height', String(selectedRoi.height))

      const response = await fetch(`${API_URL}/api/skeleton-overlay`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Could not generate the skeleton overlay (HTTP ${response.status}).`)
      }

      const blob = await response.blob()
      if (!blob.size || !blob.type.startsWith('video/')) {
        throw new Error('The backend returned an invalid overlay video. Please check the backend response.')
      }

      const overlayUrl = URL.createObjectURL(blob)
      setSkeletonOverlayUrl(overlayUrl)
      setSkeletonOverlayInfo({
        poseDetections: response.headers.get('X-Pose-Detections'),
        processedFrames: response.headers.get('X-Processed-Frames'),
      })
    } catch (error) {
      setSkeletonOverlayError(error instanceof Error ? error.message : 'Could not generate the skeleton overlay.')
    } finally {
      setIsSkeletonOverlayLoading(false)
    }
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

    if (!selectedRoi || selectedRoi.width < 0.01 || selectedRoi.height < 0.01) {
      setMessage('Please draw a selection box around the player in the first video frame.')
      return
    }

    setIsAnalysing(true)
    setMessage('Processing the selected player with CSRT tracking, MediaPipe and the LSTM autoencoder…')
    setAnalysisResult(null)
    setSkeletonOverlayUrl('')
    setSkeletonOverlayError('')
    setSkeletonOverlayInfo(null)

    try {
      const formData = new FormData()
      formData.append('file', video)
      formData.append('stroke_type', selectedStroke)
      formData.append('save_analysis', String(Boolean(user && saveAnalysis)))
      formData.append('save_video', String(Boolean(user && saveAnalysis && saveVideo)))
      formData.append('roi_x', String(selectedRoi.x))
      formData.append('roi_y', String(selectedRoi.y))
      formData.append('roi_width', String(selectedRoi.width))
      formData.append('roi_height', String(selectedRoi.height))

      const response = await fetch(`${API_URL}/api/analyse`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      })

      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || `Analysis failed (HTTP ${response.status}).`)
      }

      setAnalysisResult(data)
      if (data.saved) {
        setMessage(data.video_saved
          ? 'Analysis saved successfully. Your uploaded video is also stored.'
          : 'Analysis saved successfully.')
      } else {
        setMessage(data.warning || 'Analysis completed successfully. This result was not saved.')
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unknown error.'
      if (detail.includes('Failed to fetch') || detail.includes('NetworkError')) {
        setMessage('Cannot reach the backend. Check that FastAPI is running at http://localhost:8000 and try again.')
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
            onClick={loadHistory}
          >
            <span>◷</span> Analysis History
          </button>
          <button
            className={`nav-item ${page === 'Compare Analyses' ? 'active' : ''}`}
            onClick={loadComparisonHistory}
          >
            <span>⇄</span> Compare Analyses
          </button>
        </nav>

        <div className="sidebar-bottom">
          {!user && (
            <div className="guest-card">
              <span className="guest-icon">♙</span>
              <strong>Guest mode</strong>
              <p>Analyse strokes without an account.</p>
              <button onClick={() => setPage('Account')}>
                Create free account ↗
              </button>
            </div>
          )}
          {user && (
            <button className="sidebar-user-card" onClick={() => setPage('Account')}>
              <span className="sidebar-user-avatar">
                {user.username?.charAt(0).toUpperCase() || 'R'}
              </span>
              <span className="sidebar-user-copy">
                <strong>{user.username}</strong>
                <small>Signed in</small>
              </span>
              <span className="sidebar-user-arrow">↗</span>
            </button>
          )}
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
              <span className="status-dot" />
              {user ? user.username : 'Guest session'}
            </span>
            <button
              className="avatar"
              title="Account"
              onClick={() => setPage('Account')}
            >
              {user ? user.username.charAt(0).toUpperCase() : 'G'}
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

            {user && (
              <section className="progress-dashboard">
                <div className="section-heading progress-heading">
                  <div>
                    <p className="eyebrow">YOUR TRAINING DATA</p>
                    <h2>Player progress</h2>
                    <p>Track your saved evaluations and form scores over time.</p>
                  </div>
                  <button
                    className="secondary-button"
                    onClick={async () => {
                      setDashboardLoading(true)
                      setDashboardError('')
                      try {
                        const response = await fetch(`${API_URL}/api/analyses?limit=100`, { credentials: 'include' })
                        const data = await response.json().catch(() => ({}))
                        if (!response.ok) throw new Error(data.detail || `Could not refresh progress (HTTP ${response.status}).`)
                        setHistoryItems(Array.isArray(data.items) ? data.items : [])
                      } catch (error) {
                        setDashboardError(error.message || 'Could not refresh progress.')
                      } finally {
                        setDashboardLoading(false)
                      }
                    }}
                    disabled={dashboardLoading}
                  >
                    {dashboardLoading ? 'Refreshing…' : 'Refresh progress'}
                  </button>
                </div>

                {dashboardError && <div className="notice">{dashboardError}</div>}
                {dashboardLoading && historyItems.length === 0 ? (
                  <section className="panel progress-empty">Loading your saved evaluations…</section>
                ) : historyItems.length === 0 ? (
                  <section className="panel progress-empty">
                    <h3>Your progress starts here.</h3>
                    <p>Save your first stroke evaluation to see your performance trends and stroke breakdown.</p>
                    <button className="secondary-button" onClick={() => setPage('Analyse Stroke')}>Analyse your first stroke →</button>
                  </section>
                ) : (
                  <>
                    <div className="progress-stat-grid">
                      <article className="panel progress-stat-card">
                        <span className="progress-stat-label">SAVED ANALYSES</span>
                        <strong>{historyItems.length}</strong>
                        <small>Evaluations in your history</small>
                      </article>
                      <article className="panel progress-stat-card">
                        <span className="progress-stat-label">AVERAGE FORM SCORE</span>
                        <strong>{averageFormScore === null ? '—' : averageFormScore.toFixed(1)}</strong>
                        <small>Across {validFormAnalyses.length} scored evaluations</small>
                      </article>
                      <article className="panel progress-stat-card">
                        <span className="progress-stat-label">BEST FORM SCORE</span>
                        <strong>{bestFormScore === null ? '—' : bestFormScore.toFixed(1)}</strong>
                        <small>Your highest saved score</small>
                      </article>
                      <article className="panel progress-stat-card">
                        <span className="progress-stat-label">AVERAGE BIOMECHANICAL</span>
                        <strong>{averageScore('biomechanical_score') === null ? '—' : averageScore('biomechanical_score').toFixed(1)}</strong>
                        <small>Movement quality score</small>
                      </article>
                    </div>

                    <div className="progress-chart-grid">
                      <section className="panel progress-chart-panel">
                        <div className="panel-heading">
                          <div><h3>Form score over time</h3><p>Each point represents one saved evaluation.</p></div>
                          <span className="muted-tag">0–100</span>
                        </div>
                        {chartPoints.length > 0 ? (
                          <div className="progress-chart-wrap">
                            <svg className="progress-chart" viewBox="0 0 600 220" role="img" aria-label="Form score trend chart">
                              {[0, 25, 50, 75, 100].map((tick) => {
                                const y = 174 - (tick / 100) * 140
                                return <g key={tick}><line x1="36" x2="564" y1={y} y2={y} className="chart-grid-line" /><text x="25" y={y + 4} textAnchor="end" className="chart-axis-label">{tick}</text></g>
                              })}
                              {chartPoints.length > 1 && <polyline points={chartPoints.map((point) => `${point.x},${point.y}`).join(' ')} className="chart-trend-line" />}
                              {chartPoints.map((point, index) => <g key={`${point.date || 'score'}-${index}`}><circle cx={point.x} cy={point.y} r="5" className="chart-point"><title>{point.score.toFixed(1)} · {point.date ? new Date(point.date).toLocaleDateString() : 'Date unavailable'}</title></circle></g>)}
                            </svg>
                            <div className="chart-date-range"><span>{validFormAnalyses[0]?.created_at ? new Date(validFormAnalyses[0].created_at).toLocaleDateString() : 'Oldest'}</span><span>{validFormAnalyses[validFormAnalyses.length - 1]?.created_at ? new Date(validFormAnalyses[validFormAnalyses.length - 1].created_at).toLocaleDateString() : 'Latest'}</span></div>
                          </div>
                        ) : <p className="progress-no-data">No form scores are available yet.</p>}
                      </section>

                      <section className="panel progress-chart-panel">
                        <div className="panel-heading">
                          <div><h3>Performance by stroke</h3><p>Average form score for each category.</p></div>
                        </div>
                        <div className="stroke-progress-list">
                          {strokeProgress.map((stroke) => (
                            <div className="stroke-progress-row" key={stroke.name}>
                              <div className="stroke-progress-top"><strong>{stroke.name}</strong><span>{stroke.average === null ? '—' : stroke.average.toFixed(1)} <small>({stroke.count})</small></span></div>
                              <div className="stroke-progress-track"><span style={{ width: `${stroke.average === null ? 0 : Math.max(0, Math.min(100, stroke.average))}%` }} /></div>
                            </div>
                          ))}
                        </div>
                        <div className="progress-secondary-metrics">
                          <span>Average temporal score</span>
                          <strong>{averageScore('temporal_similarity_score') === null ? '—' : averageScore('temporal_similarity_score').toFixed(1)}</strong>
                        </div>
                      </section>
                    </div>
                    <div className="progress-footer"><span>Based on your latest {historyItems.length} saved evaluations (up to 100).</span><button className="text-button" onClick={loadHistory}>View full analysis history →</button></div>
                  </>
                )}
              </section>
            )}

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
                    ['02', 'Select player', 'Draw a box around the player'],
                    ['03', 'Evaluate form', 'Review biomechanical feedback'],
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
          <section className={`page-content inner-page upload-page ${analysisResult ? 'has-analysis-results' : ''}`}>
            <div className="upload-page-heading">
              <p className="eyebrow">VIDEO-BASED PERFORMANCE REVIEW</p>
              <h1>Analyse your stroke<span>.</span></h1>
              <p className="subheading">
                Upload your footage, select the player, and prepare for analysis.
              </p>
            </div>

            <div className="upload-layout">
              <section className="upload-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Upload your video</h2>
                    <p>Choose a recording from your device.</p>
                  </div>
                  <span className="muted-tag">{isSessionLoading ? "CHECKING SESSION" : user ? "ACCOUNT ACCESS" : "GUEST ACCESS"}</span>
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
                    <div
                      ref={roiStageRef}
                      className="roi-selection-stage"
                      onPointerDown={handleRoiPointerDown}
                      onPointerMove={handleRoiPointerMove}
                      onPointerUp={handleRoiPointerUp}
                      onPointerCancel={handleRoiPointerUp}
                      role="group"
                      aria-label="Draw a selection box around the player in the first video frame"
                    >
                      <video
                        className="video-preview roi-video-preview"
                        src={videoUrl}
                        muted
                        playsInline
                        preload="metadata"
                        onLoadedMetadata={(event) => {
                          event.currentTarget.pause()
                          if (event.currentTarget.currentTime !== 0) {
                            event.currentTarget.currentTime = 0
                          }
                        }}
                        onLoadedData={(event) => event.currentTarget.pause()}
                      />
                      <div className="roi-selection-overlay">
                        {selectedRoi && (
                          <div
                            className={`roi-selection-box ${isDrawingRoi ? 'drawing' : 'confirmed'}`}
                            style={{
                              left: `${selectedRoi.x * 100}%`,
                              top: `${selectedRoi.y * 100}%`,
                              width: `${selectedRoi.width * 100}%`,
                              height: `${selectedRoi.height * 100}%`,
                            }}
                          >
                            <span>{isDrawingRoi ? 'Selecting player…' : 'Selected player'}</span>
                          </div>
                        )}
                      </div>
                      <div className="roi-instruction">
                        {selectedRoi && !isDrawingRoi
                          ? 'Selection set · Draw again to adjust'
                          : 'Click and drag to select the player'}
                      </div>
                    </div>
                    <div className="roi-selection-controls">
                      <div className="roi-selection-copy">
                        <strong>
                          {selectedRoi && !isDrawingRoi && selectedRoi.width >= 0.01 && selectedRoi.height >= 0.01
                            ? 'Player selection ready'
                            : 'Select the player'}
                        </strong>

                        <p>
                          Draw a box around the player's full visible body in the first frame.
                        </p>
                      </div>
                    </div>

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
                      <h2>Select form reference stroke</h2>
                      <p>Choose the expected stroke for class-specific form evaluation. The LSTM classifier predicts independently.</p>
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

                <section className="save-options">
                  <div className="panel-heading">
                    <div>
                      <h2>Save your analysis</h2>
                      <p>{user ? 'Choose what you want to keep in your account.' : 'Log in to save results and review them later.'}</p>
                    </div>
                    <span className="muted-tag">{user ? 'ACCOUNT' : 'GUEST'}</span>
                  </div>
                  {user ? (
                    <>
                      <label className="save-option">
                        <input
                          type="checkbox"
                          checked={saveAnalysis}
                          onChange={(event) => {
                            setSaveAnalysis(event.target.checked)
                            if (!event.target.checked) setSaveVideo(false)
                          }}
                        />
                        <span>
                          <strong>Save analysis results</strong>
                          <small>Store your scores, feedback and evaluation metrics in Analysis History.</small>
                        </span>
                      </label>
                      <label className={`save-option ${!saveAnalysis ? 'disabled' : ''}`}>
                        <input
                          type="checkbox"
                          checked={saveVideo}
                          disabled={!saveAnalysis}
                          onChange={(event) => setSaveVideo(event.target.checked)}
                        />
                        <span>
                          <strong>Keep uploaded video</strong>
                          <small>Store a copy of the original video in secure cloud storage for a limited retention period.</small>
                        </span>
                      </label>
                    </>
                  ) : (
                    <button type="button" className="text-button" onClick={() => setPage('Account')}>
                      Log in or create an account →
                    </button>
                  )}
                </section>

                {message && (
                  <div className="notice" role="status" aria-live="polite">
                    <span>ⓘ</span>
                    {message}
                  </div>
                )}

                {analysisResult?.classification && (
                  <section className="panel analysis-results classification-results" aria-live="polite">
                    <div className="panel-heading">
                      <div>
                        <p className="eyebrow">LSTM CLASSIFIER · RUN 059</p>
                        <h2>Stroke classification</h2>
                        <p>Predicted independently from the uploaded video.</p>
                      </div>
                      <span className="muted-tag">COMPLETED</span>
                    </div>

                    <div className="form-score-card">
                      <div>
                        <span className="result-label">Predicted stroke</span>
                        <div className="form-score-value">
                          {analysisResult.classification.predicted_stroke}
                        </div>
                      </div>
                      <span className="form-rating">
                        {(Number(analysisResult.classification.confidence ?? 0) * 100).toFixed(1)}% confidence
                      </span>
                    </div>

                    <div className="classification-probabilities">
                      <h3>Class probability distribution</h3>
                      {[
                        ['clear', 'Clear'],
                        ['drop', 'Drop Shot'],
                        ['smash', 'Smash'],
                      ].map(([key, label]) => {
                        const probability = Number(analysisResult.classification.probabilities?.[key] ?? 0)
                        return (
                          <div className="classification-probability-row" key={key}>
                            <div className="classification-probability-heading">
                              <strong>{label}</strong>
                              <span>{(probability * 100).toFixed(1)}%</span>
                            </div>
                            <div className="classification-probability-track">
                              <span style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} />
                            </div>
                          </div>
                        )
                      })}
                    </div>

                    <div className="result-meta">
                      <span>Selected form reference</span>
                      <strong>{analysisResult.selected_stroke_type}</strong>
                      <span>Prediction matches selection</span>
                      <strong>{analysisResult.classification_matches_selected_stroke ? 'Yes' : 'No'}</strong>
                      <span>Model test accuracy</span>
                      <strong>{(Number(analysisResult.classification.test_accuracy ?? 0) * 100).toFixed(2)}%</strong>
                      <span>Test samples</span>
                      <strong>{analysisResult.classification.test_samples ?? '—'}</strong>
                    </div>
                    <p className="classification-note">
                      Test accuracy is the model's offline evaluation result; confidence is the probability assigned to this individual video. The selected form reference remains separate from the automatic prediction.
                    </p>
                  </section>
                )}

                {analysisResult && (
                  <section className="panel analysis-results" aria-live="polite">
                    <div className="panel-heading">
                      <div>
                        <p className="eyebrow">LSTM AUTOENCODER RESULTS</p>
                        <h2>Stroke form evaluation</h2>
                      </div>
                      <span className="muted-tag">COMPLETED</span>
                    </div>

                    <div className="report-download-row">
                      <button
                        type="button"
                        className="secondary-button report-download-button"
                        onClick={() => downloadAnalysisReport(
                          analysisResult,
                          analysisResult.analysis_id || null
                        )}
                        disabled={isReportLoading}
                      >
                        {isReportLoading ? 'Preparing PDF…' : 'Download analysis report'}
                        {!isReportLoading && <span aria-hidden="true">↓</span>}
                      </button>
                      {reportError && <p className="report-download-error" role="alert">{reportError}</p>}
                    </div>

                    {analysisResult.warning && (
                      <div className="notice" role="status">
                        {analysisResult.warning}
                      </div>
                    )}

                    <div className="form-score-card">
                      <div>
                        <span className="result-label">Overall form score</span>
                        <div className="form-score-value">
                          {Number(analysisResult.form_analysis?.form_score ?? 0).toFixed(2)}
                          <small> / 100</small>
                        </div>
                      </div>
                      <span className="form-rating">
                        {analysisResult.form_analysis?.form_rating || 'Not available'}
                      </span>
                    </div>

                    <div className="form-metric-grid">
                      <article className="form-metric-card">
                        <span>Biomechanical score</span>
                        <strong>
                          {Number(analysisResult.form_analysis?.biomechanical_score ?? 0).toFixed(2)}
                        </strong>
                        <small> / 100</small>
                      </article>
                      <article className="form-metric-card">
                        <span>Temporal similarity</span>
                        <strong>
                          {Number(analysisResult.form_analysis?.temporal_similarity_score ?? 0).toFixed(2)}
                        </strong>
                        <small> / 100</small>
                      </article>
                      <article className="form-metric-card">
                        <span>Reconstruction error</span>
                        <strong>
                          {Number(analysisResult.form_analysis?.reconstruction_error ?? 0).toFixed(6)}
                        </strong>
                      </article>
                    </div>

                    <div className="result-meta">
                      <span>Selected stroke</span>
                      <strong>{analysisResult.selected_stroke_type}</strong>
                      <span>Frames processed</span>
                      <strong>{analysisResult.video_info?.frame_count ?? '—'}</strong>
                      <span>Pose detection</span>
                      <strong>
                        {Math.round((analysisResult.video_info?.pose_detection_rate ?? 0) * 100)}%
                      </strong>
                      <span>Average landmark visibility</span>
                      <strong>
                        {Math.round((analysisResult.video_info?.visibility_mean ?? 0) * 100)}%
                      </strong>
                      <span>Tracker failures</span>
                      <strong>{analysisResult.video_info?.tracker_failures ?? '—'}</strong>
                    </div>

                    <section className="skeleton-overlay-section">
                      <div className="skeleton-overlay-copy">
                        <p className="eyebrow">VISUAL MOVEMENT TRACKING</p>
                        <h3>Skeleton overlay video</h3>
                        <p>
                          View your uploaded footage with MediaPipe's detected body joints
                          and connections drawn over the player throughout the movement.
                        </p>
                      </div>

                      {!skeletonOverlayUrl && (
                        <button
                          type="button"
                          className="primary-button skeleton-overlay-button"
                          onClick={generateSkeletonOverlay}
                          disabled={isSkeletonOverlayLoading}
                        >
                          {isSkeletonOverlayLoading ? 'Generating overlay…' : 'Generate skeleton overlay'}
                          {!isSkeletonOverlayLoading && <span>↗</span>}
                        </button>
                      )}

                      {isSkeletonOverlayLoading && (
                        <div className="skeleton-overlay-status" role="status">
                          Processing each video frame and drawing the detected body landmarks.
                          This may take a little while for longer videos.
                        </div>
                      )}

                      {skeletonOverlayError && (
                        <div className="notice skeleton-overlay-error" role="alert">
                          {skeletonOverlayError}
                        </div>
                      )}

                      {skeletonOverlayUrl && (
                        <div className="skeleton-overlay-result">
                          <div className="skeleton-overlay-video-heading">
                            <strong>Annotated movement video</strong>
                            <button
                              type="button"
                              className="text-button"
                              onClick={() => {
                                setSkeletonOverlayUrl('')
                                setSkeletonOverlayInfo(null)
                                setSkeletonOverlayError('')
                              }}
                            >
                              Close overlay
                            </button>
                          </div>
                          <video
                            className="skeleton-overlay-player"
                            src={skeletonOverlayUrl}
                            controls
                            playsInline
                            preload="metadata"
                          >
                            Your browser does not support HTML video playback.
                          </video>
                          <div className="skeleton-overlay-meta">
                            {skeletonOverlayInfo?.processedFrames && (
                              <span>{skeletonOverlayInfo.processedFrames} frames processed</span>
                            )}
                            {skeletonOverlayInfo?.poseDetections && (
                              <span>{skeletonOverlayInfo.poseDetections} pose detections</span>
                            )}
                          </div>
                          <p className="skeleton-overlay-caption">
                            The original uploaded video remains unchanged. This is a separate
                            annotated copy generated for visual review.
                          </p>
                        </div>
                      )}
                    </section>

                    {(() => {
                      const detail = analysisResult.form_analysis?.detailed_feedback || {}
                      const measurements = Array.isArray(detail.all_measurements)
                        ? detail.all_measurements
                        : [
                            ...(detail.joint_angles || []),
                            ...(detail.torso_alignment || []),
                            ...(detail.elbow_angle_changes || []),
                            ...(detail.angular_velocity || []),
                            ...(detail.angular_acceleration || []),
                            ...(detail.movement_timing || []),
                          ]

                      const reviewItems = Array.isArray(detail.areas_to_review)
                        ? detail.areas_to_review
                        : []

                      const getFriendlyFeedback = (item) => {
                        const feature = String(item.feature || '').toLowerCase()
                        if (feature.includes('peak-angle timing')) {
                          if (feature.includes('elbow')) {
                            return {
                              title: 'Improve your arm timing',
                              message: `Your ${feature.includes('left') ? 'left' : 'right'} arm reaches its peak position a little too late. Try to coordinate your arm movement earlier.`,
                            }
                          }
                          if (feature.includes('shoulder')) {
                            return {
                              title: 'Improve your shoulder timing',
                              message: `Your ${feature.includes('left') ? 'left' : 'right'} shoulder reaches its peak position a little too late. Try to coordinate your shoulder movement earlier.`,
                            }
                          }
                          if (feature.includes('hip')) {
                            return {
                              title: 'Work on your hip movement',
                              message: `Your ${feature.includes('left') ? 'left' : 'right'} hip movement happens a little too late. Focus on coordinating your body rotation with your swing.`,
                            }
                          }
                          if (feature.includes('knee')) {
                            return {
                              title: 'Improve your leg timing',
                              message: `Your ${feature.includes('left') ? 'left' : 'right'} leg reaches its peak position a little too late. Try to coordinate your leg movement with your swing.`,
                            }
                          }
                        }
                        if (feature.includes('angular-velocity') || feature.includes('angular velocity')) {
                          return {
                            title: 'Work on your swing timing',
                            message: `Your ${feature.includes('left') ? 'left' : 'right'} arm movement speed differs from the movement pattern detected. Practise a smooth, controlled swing.`,
                          }
                        }
                        if (feature.includes('torso') || feature.includes('lean')) {
                          return {
                            title: 'Improve your body balance',
                            message: 'Your upper body leans more to one side during the movement. Focus on staying balanced as you prepare and swing.',
                          }
                        }
                        if (feature.includes('knee') || feature.includes('leg')) {
                          return {
                            title: 'Work on your leg positioning',
                            message: 'Pay attention to your leg position during the stroke and try to keep your movement controlled and balanced.',
                          }
                        }
                        if (feature.includes('hip')) {
                          return {
                            title: 'Work on your hip movement',
                            message: 'Focus on coordinating your hip movement with your upper body as you swing.',
                          }
                        }
                        if (feature.includes('elbow') || feature.includes('shoulder') || feature.includes('arm')) {
                          return {
                            title: 'Work on your arm positioning',
                            message: 'Pay attention to your arm position throughout the swing and practise moving smoothly.',
                          }
                        }
                        return {
                          title: 'Keep practising this movement',
                          message: 'Focus on performing this part of the stroke smoothly and with control.',
                        }
                      }

                      const improvementItems = reviewItems
                        .slice()
                        .sort((a, b) => Number(b.deviation || 0) - Number(a.deviation || 0))
                        .slice(0, 3)
                        .map(getFriendlyFeedback)

                      const strengthFeatures = (Array.isArray(detail.strengths) ? detail.strengths : [])
                        .map((item) => String(item.feature || '').toLowerCase())
                      const hasArmStrength = strengthFeatures.some((feature) =>
                        feature.includes('elbow') || feature.includes('shoulder')
                      )
                      const hasLegStrength = strengthFeatures.some((feature) =>
                        feature.includes('hip') || feature.includes('knee')
                      )
                      const hasTorsoStrength = strengthFeatures.some((feature) =>
                        feature.includes('torso')
                      )

                      const strengths = []
                      if (hasArmStrength) {
                        strengths.push({
                          title: 'Good arm positioning',
                          message: 'Your arm positioning is looking good. Keep it up!',
                        })
                      }
                      if (hasLegStrength) {
                        strengths.push({
                          title: 'Good lower-body positioning',
                          message: 'Your hip and leg positioning are looking good. Keep working on it!',
                        })
                      }
                      if (hasTorsoStrength) {
                        strengths.push({
                          title: 'Good body control',
                          message: 'Your body posture is staying close to the movement pattern. Keep it up!',
                        })
                      }
                      const visibleStrengths = strengths.slice(0, 2)

                      let trainingTip = 'Practise your stroke slowly and focus on smooth, controlled movements before increasing your speed.'
                      if (improvementItems.some((item) => /timing|hip movement|leg timing/i.test(item.title))) {
                        trainingTip = 'Practise your overhead swing slowly, focusing on coordinating your arm and body movement before increasing your speed.'
                      } else if (improvementItems.some((item) => /balance|positioning/i.test(item.title))) {
                        trainingTip = 'Practise the movement at a comfortable pace. Keep your body balanced and your movements controlled throughout the stroke.'
                      }

                      return (
                        <section className="form-analysis-section player-feedback-section">
                          <div className="panel-heading">
                            <div>
                              <p className="eyebrow">PERSONALISED OBSERVATIONS</p>
                              <h2>Your movement feedback</h2>
                              <p className="detailed-feedback-intro">Simple tips to help you understand and improve your stroke.</p>
                            </div>
                          </div>

                          {visibleStrengths.length > 0 && (
                            <div className="player-feedback-group">
                              <h3 className="player-feedback-heading strength-heading">✓ What you're doing well</h3>
                              <div className="player-feedback-grid">
                                {visibleStrengths.map((item, index) => (
                                  <article className="player-feedback-card strength-card" key={`strength-${index}`}>
                                    <strong>{item.title}</strong>
                                    <p>{item.message}</p>
                                  </article>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="player-feedback-group">
                            <h3 className="player-feedback-heading improve-heading">◎ What to improve</h3>
                            {improvementItems.length > 0 ? (
                              <div className="player-feedback-grid">
                                {improvementItems.map((item, index) => (
                                  <article className="player-feedback-card improve-card" key={`improve-${index}`}>
                                    <strong>{item.title}</strong>
                                    <p>{item.message}</p>
                                  </article>
                                ))}
                              </div>
                            ) : (
                              <article className="player-feedback-card strength-card">
                                <strong>Keep up the good work</strong>
                                <p>No major movement areas were flagged in this analysis. Keep practising to build consistency.</p>
                              </article>
                            )}
                          </div>

                          <div className="player-training-tip">
                            <span className="training-tip-icon">✦</span>
                            <div>
                              <strong>Training tip</strong>
                              <p>{trainingTip}</p>
                            </div>
                          </div>

                          <div className="technical-breakdown">
                            <button
                              type="button"
                              className="technical-breakdown-toggle"
                              onClick={() => setShowTechnicalBreakdown((current) => !current)}
                              aria-expanded={showTechnicalBreakdown}
                            >
                              <span>
                                <strong>View technical breakdown</strong>
                                <small>Joint angles, movement timing and other measurements</small>
                              </span>
                              <span className="technical-breakdown-chevron">{showTechnicalBreakdown ? '−' : '+'}</span>
                            </button>

                            {showTechnicalBreakdown && (
                              <div className="technical-breakdown-content">
                                {detail.summary && <p className="detailed-feedback-summary">{detail.summary}</p>}
                                {measurements.length > 0 ? (
                                  <div className="detailed-feedback-grid">
                                    {measurements.map((measurement, index) => {
                                      const label = measurement.name || measurement.feature || measurement.metric || measurement.label || `Measurement ${index + 1}`
                                      const value = measurement.value ?? measurement.player_value ?? measurement.actual_value
                                      const reference = measurement.reference ?? measurement.reference_value ?? measurement.target_value
                                      const deviation = measurement.deviation ?? measurement.z_score ?? measurement.difference
                                      const unit = measurement.unit || ''
                                      const status = measurement.status_label || measurement.status || measurement.level || measurement.rating
                                      return (
                                        <article className="detailed-feedback-card" key={`${label}-${index}`}>
                                          <div className="detailed-feedback-card-top">
                                            <strong>{String(label).replaceAll('_', ' ')}</strong>
                                            {status && <span className="detailed-feedback-status">{String(status).replaceAll('_', ' ')}</span>}
                                          </div>
                                          <div className="detailed-feedback-values">
                                            <span><small>Your measurement</small><strong>{value == null ? '—' : `${value} ${unit}`}</strong></span>
                                            <span><small>Comparison value</small><strong>{reference == null ? '—' : `${reference} ${unit}`}</strong></span>
                                            <span><small>Deviation</small><strong>{deviation == null ? '—' : Number.isFinite(Number(deviation)) ? Number(deviation).toFixed(3) : String(deviation)}</strong></span>
                                          </div>
                                          {measurement.explanation && <p className="detailed-feedback-note">{measurement.explanation}</p>}
                                        </article>
                                      )
                                    })}
                                  </div>
                                ) : (
                                  <p className="form-analysis-intro">Detailed measurement data was not included in this response.</p>
                                )}
                                {detail.interpretation && <p className="detailed-feedback-summary">{detail.interpretation}</p>}
                              </div>
                            )}
                          </div>
                        </section>
                      )
                    })()}

                    <p className="result-disclaimer">
                      {analysisResult.form_analysis?.score_interpretation ||
                        'Prototype similarity score based on reference statistics and autoencoder reconstruction.'}
                    </p>
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
                  disabled={isAnalysing || !video || !selectedRoi || selectedRoi.width < 0.01 || selectedRoi.height < 0.01 || isDrawingRoi}
                >
                  {isAnalysing ? 'Analysing video…' : 'Analyse video'} <span>{isAnalysing ? '⏳' : '→'}</span>
                </button>

                <p className="privacy-note">
                  {user
                    ? (saveAnalysis
                      ? (saveVideo ? 'Your results and uploaded video will be saved to your account.' : 'Your results will be saved; the uploaded video will not be retained.')
                      : 'Your analysis will be temporary and will not be saved.')
                    : 'Guest mode · Your analysis and uploaded video are temporary and are not saved to an account.'}
                </p>
              </section>

              <aside className="upload-side-panel">
                <div className="upload-side-icon">◎</div>
                <p className="eyebrow">YOUR ANALYSIS</p>
                <h2>From footage to feedback.</h2>
                <p>
                  Select the player in your first video frame before pose tracking and stroke
                  form evaluation.
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
                    <strong>Select player</strong>
                    <small>{selectedRoi && !isDrawingRoi ? 'Player selection ready' : 'Draw a box on the first frame'}</small>
                  </div>
                  {selectedRoi && !isDrawingRoi && <span className="step-complete">✓</span>}
                </div>

                <div className="upload-progress-step">
                  <span>04</span>
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
          <section className="page-content inner-page history-page">
            <p className="eyebrow">YOUR ACTIVITY</p>
            <h1>Analysis history<span>.</span></h1>
            <p className="subheading">Review the stroke evaluations you chose to save.</p>

            {!user ? (
              <section className="empty-state panel">
                <span className="empty-icon">♙</span>
                <h2>Log in to view your history</h2>
                <p>Saved analyses are linked to your account. Log in to access them.</p>
                <button className="primary-button" onClick={() => setPage('Account')}>Go to account →</button>
              </section>
            ) : isHistoryLoading ? (
              <section className="empty-state panel"><h2>Loading your analyses…</h2></section>
            ) : historyError ? (
              <section className="panel">
                <div className="notice">{historyError}</div>
                <button className="secondary-button" onClick={loadHistory}>Try again</button>
              </section>
            ) : historyItems.length === 0 ? (
              <section className="empty-state panel">
                <span className="empty-icon">◷</span>
                <h2>No saved analyses yet</h2>
                <p>When you choose “Save analysis results” after an evaluation, it will appear here.</p>
                <button className="primary-button" onClick={() => setPage('Analyse Stroke')}>Analyse a stroke →</button>
              </section>
            ) : (
              <>
                <div className="history-toolbar">
                  <span>{historyItems.length} saved {historyItems.length === 1 ? 'analysis' : 'analyses'}</span>
                  <button className="secondary-button" onClick={loadHistory}>Refresh</button>
                </div>
                <div className="history-list">
                  {historyItems.map((item) => (
                    <article className="history-card panel" key={item.id}>
                      <div className="history-card-main">
                        <div>
                          <span className="history-date">{item.created_at ? new Date(item.created_at).toLocaleString() : 'Date unavailable'}</span>
                          <h2>{item.stroke_type || 'Stroke analysis'}</h2>
                          <p>{item.video_filename || 'Saved analysis results'}</p>
                        </div>
                        <div className="history-score">
                          <strong>{item.form_score == null ? '—' : Number(item.form_score).toFixed(1)}</strong>
                          <small>FORM SCORE</small>
                        </div>
                      </div>
                      <div className="history-metrics">
                        <span>Biomechanical <strong>{item.biomechanical_score == null ? '—' : Number(item.biomechanical_score).toFixed(1)}</strong></span>
                        <span>Temporal <strong>{item.temporal_similarity_score == null ? '—' : Number(item.temporal_similarity_score).toFixed(1)}</strong></span>
                        <span>Video <strong>{item.video_saved ? 'Retained' : 'Not saved'}</strong></span>
                      </div>
                      <div className="history-card-actions">
                        <button className="text-button" onClick={() => openHistoryItem(item)}>View analysis details →</button>
                        {item.video_saved && (
                          <button
                            className="text-button"
                            onClick={() => openHistoryVideo(item)}
                            disabled={isHistoryVideoLoading}
                          >
                            {isHistoryVideoLoading ? 'Preparing video…' : 'Watch original video →'}
                          </button>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
                {historyVideoError && (
                  <section className="panel history-video-panel">
                    <div className="notice">{historyVideoError}</div>
                  </section>
                )}
                {historyVideo && (
                  <section className="panel history-video-panel">
                    <div className="panel-heading">
                      <div>
                        <p className="eyebrow">SAVED ORIGINAL VIDEO</p>
                        <h2>{historyVideo.filename}</h2>
                      </div>
                      <button className="text-button" onClick={() => setHistoryVideo(null)}>Close video</button>
                    </div>
                    <video
                      className="history-video-player"
                      src={historyVideo.url}
                      controls
                      playsInline
                      preload="metadata"
                    >
                      Your browser does not support HTML video playback.
                    </video>
                    <p className="form-analysis-intro">
                      This private playback link expires after 15 minutes. Reopen the video from history to generate a fresh link.
                    </p>
                  </section>
                )}
                {selectedHistory && (
                  <section className="panel history-detail">
                    <div className="panel-heading">
                      <div><p className="eyebrow">SAVED EVALUATION</p><h2>Analysis details</h2></div>
                      <div className="history-detail-actions">
                        <button
                          type="button"
                          className="secondary-button report-download-button"
                          onClick={() => downloadAnalysisReport(
                            selectedHistory.result ?? selectedHistory,
                            selectedHistory.id || null
                          )}
                          disabled={isReportLoading}
                        >
                          {isReportLoading ? 'Preparing PDF…' : 'Download report'}
                        </button>
                        <button className="text-button" onClick={() => setSelectedHistory(null)}>Close</button>
                      </div>
                    </div>
                    {reportError && <p className="report-download-error" role="alert">{reportError}</p>}
                    <pre>{JSON.stringify(selectedHistory.result ?? selectedHistory, null, 2)}</pre>
                  </section>
                )}
              </>
            )}
          </section>
        )}


        {page === 'Compare Analyses' && (
          <section className="page-content inner-page comparison-page">
            <p className="eyebrow">PERFORMANCE TRACKING</p>
            <h1>Compare previous analyses<span>.</span></h1>
            <p className="subheading">
              Select two to five saved sessions of the same stroke type to review how your performance changes over time.
            </p>

            {!user ? (
              <section className="empty-state panel">
                <span className="empty-icon">⇄</span>
                <h2>Log in to compare performances</h2>
                <p>Your saved analyses are linked to your account. Log in to select sessions and compare your results.</p>
                <button className="primary-button" onClick={() => setPage('Account')}>Go to account →</button>
              </section>
            ) : isHistoryLoading ? (
              <section className="empty-state panel"><h2>Loading your saved analyses…</h2></section>
            ) : comparisonError && historyItems.length === 0 ? (
              <section className="panel comparison-empty">
                <div className="notice" role="alert">{comparisonError}</div>
                <button className="secondary-button" onClick={loadComparisonHistory}>Try again</button>
              </section>
            ) : filteredComparisonCandidates.length < 2 ? (
              <section className="empty-state panel">
                <span className="empty-icon">⇄</span>
                <h2>Not enough saved analyses</h2>
                <p>You need at least two saved analyses with valid form scores before you can compare your performance.</p>
                <button className="primary-button" onClick={() => setPage('Analyse Stroke')}>Analyse a stroke →</button>
              </section>
            ) : (
              <>
                <section className="panel comparison-selector">
                  <div className="panel-heading">
                    <div>
                      <p className="eyebrow">STEP 1</p>
                      <h2>Select your sessions</h2>
                      <p>Choose 2–5 sessions. For a fair comparison, all selected sessions must use the same stroke type.</p>
                    </div>
                    <span className="comparison-count">{comparisonIds.length}/5 selected</span>
                  </div>

                  {comparisonError && <div className="notice comparison-notice" role="alert">{comparisonError}</div>}

                  <div className="comparison-session-list">
                    {comparisonCandidates.map((item) => {
                      const isSelected = comparisonIds.includes(item.id)
                      const firstSelected = historyItems.find((entry) => entry.id === comparisonIds[0])
                      const differentStroke = firstSelected &&
                        (firstSelected.stroke_type || '').toLowerCase() !== (item.stroke_type || '').toLowerCase()
                      const disabled = (!isSelected && (comparisonIds.length >= 5 || differentStroke))

                      return (
                        <label className={`comparison-session ${isSelected ? 'selected' : ''} ${disabled ? 'disabled' : ''}`} key={item.id}>
                          <input
                            type="checkbox"
                            checked={isSelected}
                            disabled={disabled}
                            onChange={() => toggleComparisonSelection(item)}
                          />
                          <span className="comparison-session-main">
                            <strong>{item.stroke_type || 'Stroke analysis'}</strong>
                            <small>{item.created_at ? new Date(item.created_at).toLocaleString() : 'Date unavailable'}</small>
                            <small>{item.video_filename || 'Saved analysis'}</small>
                          </span>
                          <span className="comparison-session-score">
                            <strong>{Number(item.form_score).toFixed(1)}</strong>
                            <small>FORM SCORE</small>
                          </span>
                        </label>
                      )
                    })}
                  </div>

                  <div className="comparison-selector-footer">
                    <span>{comparisonIds.length < 2 ? 'Select at least two sessions to compare.' : `${comparisonIds.length} sessions ready to compare.`}</span>
                    <button className="text-button" type="button" onClick={() => { setComparisonIds([]); setComparisonError('') }} disabled={comparisonIds.length === 0}>Clear selection</button>
                  </div>
                </section>

                {selectedComparisons.length >= 2 && (
                  <section className="comparison-results">
                    <div className="section-heading comparison-results-heading">
                      <div>
                        <p className="eyebrow">STEP 2</p>
                        <h2>{comparisonStroke} performance comparison</h2>
                        <p>Sessions are arranged from oldest to newest.</p>
                      </div>
                      <span className="section-count">{selectedComparisons.length} SESSIONS</span>
                    </div>

                    <div className="comparison-summary-grid">
                      {[
                        { key: 'form_score', label: 'Overall form', higher: true },
                        { key: 'biomechanical_score', label: 'Biomechanical', higher: true },
                        { key: 'temporal_similarity_score', label: 'Temporal similarity', higher: true },
                        { key: 'reconstruction_error', label: 'Reconstruction error', higher: false },
                      ].map((metric) => {
                        const firstValue = Number(selectedComparisons[0][metric.key])
                        const lastValue = Number(selectedComparisons[selectedComparisons.length - 1][metric.key])
                        const hasValues = selectedComparisons[0][metric.key] != null &&
                          selectedComparisons[selectedComparisons.length - 1][metric.key] != null &&
                          Number.isFinite(firstValue) && Number.isFinite(lastValue)
                        const change = hasValues ? lastValue - firstValue : null
                        const improved = hasValues && (metric.higher ? change > 0 : change < 0)
                        const declined = hasValues && (metric.higher ? change < 0 : change > 0)
                        return (
                          <article className="panel comparison-summary-card" key={metric.key}>
                            <span>{metric.label}</span>
                            <strong>{hasValues ? `${change > 0 ? '+' : ''}${change.toFixed(2)}` : '—'}</strong>
                            <small className={improved ? 'comparison-improved' : declined ? 'comparison-declined' : ''}>
                              {!hasValues ? 'Insufficient data' : improved ? 'Improved from first to latest' : declined ? 'Declined from first to latest' : 'No change from first to latest'}
                            </small>
                          </article>
                        )
                      })}
                    </div>

                    <section className="panel comparison-table-panel">
                      <div className="panel-heading">
                        <div><p className="eyebrow">SESSION BREAKDOWN</p><h2>Metric comparison</h2></div>
                      </div>
                      <div className="comparison-table-wrap">
                        <table className="comparison-table">
                          <thead>
                            <tr>
                              <th>Metric</th>
                              {selectedComparisons.map((item, index) => (
                                <th key={item.id}>Session {index + 1}<small>{item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Date unavailable'}</small></th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {[
                              { key: 'form_score', label: 'Overall form score', suffix: '/100' },
                              { key: 'biomechanical_score', label: 'Biomechanical score', suffix: '/100' },
                              { key: 'temporal_similarity_score', label: 'Temporal similarity', suffix: '/100' },
                              { key: 'reconstruction_error', label: 'Reconstruction error', suffix: '' },
                            ].map((metric) => (
                              <tr key={metric.key}>
                                <th>{metric.label}</th>
                                {selectedComparisons.map((item) => {
                                  const value = item[metric.key]
                                  return <td key={item.id}>{value == null || !Number.isFinite(Number(value)) ? '—' : `${Number(value).toFixed(2)}${metric.suffix}`}</td>
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </section>

                    <section className="panel comparison-timeline-panel">
                      <div className="panel-heading">
                        <div><p className="eyebrow">FORM SCORE TREND</p><h2>Progress across selected sessions</h2></div>
                      </div>
                      <div className="comparison-trend-list">
                        {selectedComparisons.map((item, index) => {
                          const score = Math.max(0, Math.min(100, Number(item.form_score) || 0))
                          return (
                            <div className="comparison-trend-item" key={item.id}>
                              <div className="comparison-trend-label">
                                <span>Session {index + 1} · {item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Date unavailable'}</span>
                                <strong>{score.toFixed(1)}</strong>
                              </div>
                              <div className="comparison-trend-track"><span style={{ width: `${score}%` }} /></div>
                            </div>
                          )
                        })}
                      </div>
                    </section>
                    <p className="comparison-footnote">A higher form, biomechanical, or temporal similarity score indicates a better result. A lower reconstruction error indicates closer similarity to the reference movement. These metrics support performance tracking and are not a substitute for coaching assessment.</p>
                  </section>
                )}
              </>
            )}
          </section>
        )}


        {page === 'Account' && (
          <section className="page-content inner-page">
            <p className="eyebrow">YOUR ACCOUNT</p>

            <h1>
              {user ? 'Welcome back' : 'Keep your progress'}
              <span>.</span>
            </h1>

            <section className={`account-options panel ${user ? "account-options-authenticated" : ""}`}>

              {user ? (
                <div className="account-overview-grid">
                  <section className="account-profile">
                    <div className="account-profile-header">
                      <div className="account-avatar">
                        {user.username?.charAt(0).toUpperCase() || 'R'}
                      </div>
                      <div className="account-profile-heading">
                        <span className="account-status">
                          <span className="account-status-dot" />
                          ACCOUNT ACTIVE
                        </span>
                        <h2>{user.username}</h2>
                        <p>Manage your Rally profile and track your badminton analysis activity.</p>
                      </div>
                    </div>

                    <div className="account-section-heading">
                      <div>
                        <h3>Personal information</h3>
                        <p>Your registered account details</p>
                      </div>
                    </div>

                    <div className="account-details-grid">
                      <div className="account-detail-card">
                        <span>USERNAME</span>
                        <strong>{user.username}</strong>
                      </div>
                      <div className="account-detail-card">
                        <span>EMAIL ADDRESS</span>
                        <strong>{user.email}</strong>
                      </div>
                      <div className="account-detail-card account-id-card">
                        <span>ACCOUNT ID</span>
                        <strong>{user.id}</strong>
                      </div>
                    </div>

                    <div className="account-actions">
                      <button
                        className="primary-button"
                        onClick={() => setPage('Analyse Stroke')}
                      >
                        Analyse a stroke <span>→</span>
                      </button>
                      <button
                        className="secondary-button"
                        onClick={handleLogout}
                      >
                        Log out
                      </button>
                    </div>
                  </section>

                  <aside className="account-side-column">
                    <section className="account-activity-card">
                      <p className="eyebrow">YOUR RALLY ACTIVITY</p>
                      <h3>Analysis overview</h3>
                      <p className="account-activity-description">
                        Your saved evaluations and performance data at a glance.
                      </p>
                      <div className="account-stat-list">
                        <div className="account-stat">
                          <span>Saved analyses</span>
                          <strong>{historyItems.length}</strong>
                        </div>
                        <div className="account-stat">
                          <span>Average form score</span>
                          <strong>{averageFormScore === null ? '—' : `${averageFormScore.toFixed(1)}/100`}</strong>
                        </div>
                        <div className="account-stat">
                          <span>Stroke categories</span>
                          <strong>{strokeProgress.filter((item) => item.count > 0).length}/3</strong>
                        </div>
                      </div>
                      {dashboardLoading && <p className="account-activity-note">Refreshing your activity…</p>}
                      {dashboardError && <p className="account-activity-note">{dashboardError}</p>}
                    </section>

                    <section className="account-quick-links">
                      <p className="eyebrow">QUICK ACCESS</p>
                      <button onClick={() => setPage('History')}>
                        <span><strong>Analysis history</strong><small>Review your saved sessions</small></span>
                        <span>→</span>
                      </button>
                      <button onClick={() => setPage('Compare Analyses')}>
                        <span><strong>Compare analyses</strong><small>Track changes across sessions</small></span>
                        <span>→</span>
                      </button>
                    </section>
                  </aside>
                </div>
              ) : (
                <>
                  <div className="account-symbol">♙</div>

                  <h2>
                    {authMode === 'register'
                      ? 'Create your account'
                      : 'Welcome back'}
                  </h2>

                  <p>
                    {authMode === 'register'
                      ? 'Create an account to start saving your badminton analyses.'
                      : 'Log in to access your Rally account.'}
                  </p>

                  <form onSubmit={handleAuthSubmit}>
                    <div className="auth-form-field">
                      <label htmlFor="auth-email">Email address</label>
                      <input
                        id="auth-email"
                        type="email"
                        value={authEmail}
                        onChange={(event) =>
                          setAuthEmail(event.target.value)
                        }
                        placeholder="you@example.com"
                        required
                      />
                    </div>

                    {authMode === 'register' && (
                      <div className="auth-form-field">
                        <label htmlFor="auth-username">Username</label>
                        <input
                          id="auth-username"
                          type="text"
                          value={authUsername}
                          onChange={(event) =>
                            setAuthUsername(event.target.value)
                          }
                          placeholder="Choose a username"
                          minLength={3}
                          maxLength={100}
                          required
                        />
                      </div>
                    )}

                    <div className="auth-form-field">
                      <label htmlFor="auth-password">Password</label>
                      <input
                        id="auth-password"
                        type="password"
                        value={authPassword}
                        onChange={(event) =>
                          setAuthPassword(event.target.value)
                        }
                        placeholder="Enter your password"
                        minLength={authMode === 'register' ? 8 : undefined}
                        maxLength={128}
                        required
                      />
                    </div>

                    <button
                      type="submit"
                      className="primary-button"
                      disabled={isAuthLoading}
                    >
                      {isAuthLoading
                        ? 'Please wait...'
                        : authMode === 'register'
                          ? 'Create account →'
                          : 'Log in →'}
                    </button>
                  </form>

                  {authMessage && (
                    <div className="notice" role="status">
                      {authMessage}
                    </div>
                  )}

                  <div className="account-divider">
                    <span>OR</span>
                  </div>

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => {
                      setAuthMode(
                        authMode === 'login' ? 'register' : 'login'
                      )
                      setAuthMessage('')
                      setAuthPassword('')
                    }}
                  >
                    {authMode === 'login'
                      ? 'Create a new account'
                      : 'Already have an account? Log in'}
                  </button>

                  <button
                    type="button"
                    className="text-button"
                    onClick={() => setPage('Analyse Stroke')}
                  >
                    Continue as guest →
                  </button>
                </>
              )}

            </section>
          </section>
        )}

      </main>
    </div>
  )
}

export default App
