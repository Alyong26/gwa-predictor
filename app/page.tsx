'use client'
import { useState } from 'react'

export default function Home() {
  const predictEndpoint =
    process.env.NEXT_PUBLIC_PREDICT_ENDPOINT ||
    (typeof window !== 'undefined' && window.location.hostname === 'localhost'
      ? 'http://localhost:8000/predict'
      : '/api/predict')
  const [overallGwa, setOverallGwa] = useState('')
  const [resultGwa, setResultGwa] = useState<number | null>(null)
  const [resultInterval, setResultInterval] = useState<{ lower: number; upper: number } | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function predict() {
    setLoading(true)
    setError('')

    const payload = {
      overall_gwa: Number(overallGwa)
    }

    try {
      const res = await fetch(predictEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const data = await res.json()

      if (!res.ok) {
        setError(data.detail || 'Prediction failed.')
        setResultGwa(null)
        setResultInterval(null)
        return
      }

      setResultGwa(data.predicted_gwa)
      setResultInterval(
        data.prediction_interval
          ? {
              lower: data.prediction_interval.lower_gwa,
              upper: data.prediction_interval.upper_gwa
            }
          : null
      )
    } catch {
      setError('Could not connect to the API. Make sure FastAPI is running on port 8000.')
      setResultGwa(null)
      setResultInterval(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className='page'>
      <div className='background'>
        <div className='blob blobOne' />
        <div className='blob blobTwo' />
        <div className='blob blobThree' />
        <div className='grid' />
        <div className='glow' />
      </div>

      <section className='card'>
        <img src='/icon-512.png' alt='GWA Predictor logo' className='logo' />
        <span className='badge'>Smart Academic Forecast</span>
        <h1>GWA Predictor</h1>
        <p className='subtitle'>
          Enter your current GWA to predict your next academic year GWA.
          <br /> Developer: Al James Lopez 
          <br /> © 2026
        </p>

        <div className='heroStats'>
          <div className='statBox'>
            <span className='statLabel'>Prediction target</span>
            <strong>NEXT A.Y. GWA</strong>
          </div>
          <div className='statBox'>
            <span className='statLabel'>Input required</span>
            <strong>CURRENT GWA</strong>
          </div>
        </div>

        <div className='singleFieldWrap'>
          <label className='field'>
            <span>Overall GWA (1.00 - 5.00)</span>
            <input
              type='number'
              step='0.01'
              min='1'
              max='5'
              value={overallGwa}
              onChange={(e) => setOverallGwa(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && overallGwa && !loading) {
                  e.preventDefault()
                  predict()
                }
              }}
              placeholder='Example: 2.25'
              className='input'
            />
          </label>
        </div>

        <button
          onClick={predict}
          className='button'
          disabled={loading || !overallGwa}
        >
          {loading ? 'Predicting...' : 'Predict Next GWA'}
        </button>

        {error && <div className='errorBox'>{error}</div>}

        {resultGwa !== null && (
          <div className='result'>
            <div className='resultRow'>
              <span>Predicted next-year GWA</span>
              <strong>{resultGwa.toFixed(2)}</strong>
            </div>
            {resultInterval && <div className='resultMeta'>Likely range (80%): {resultInterval.lower.toFixed(2)} - {resultInterval.upper.toFixed(2)}</div>}
          </div>
        )}
      </section>

      <style jsx>{`
        .page {
          position: relative;
          min-height: 100vh;
          display: grid;
          place-items: center;
          overflow: hidden;
          padding: 24px;
          background:
            radial-gradient(circle at top, rgba(129, 140, 248, 0.18), transparent 30%),
            linear-gradient(135deg, #07111f 0%, #0f172a 40%, #172554 100%);
        }

        .background {
          position: absolute;
          inset: 0;
          overflow: hidden;
        }

        .blob {
          position: absolute;
          border-radius: 999px;
          filter: blur(18px);
          opacity: 0.55;
          animation: float 16s ease-in-out infinite;
        }

        .blobOne {
          top: -10%;
          left: -8%;
          width: 26rem;
          height: 26rem;
          background: radial-gradient(circle, rgba(56, 189, 248, 0.95) 0%, rgba(56, 189, 248, 0.08) 70%);
        }

        .blobTwo {
          right: -10%;
          top: 12%;
          width: 32rem;
          height: 32rem;
          background: radial-gradient(circle, rgba(168, 85, 247, 0.9) 0%, rgba(168, 85, 247, 0.08) 70%);
          animation-delay: -5s;
        }

        .blobThree {
          bottom: -14%;
          left: 22%;
          width: 28rem;
          height: 28rem;
          background: radial-gradient(circle, rgba(34, 197, 94, 0.75) 0%, rgba(34, 197, 94, 0.05) 72%);
          animation-delay: -9s;
        }

        .grid {
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.08) 1px, transparent 1px);
          background-size: 72px 72px;
          mask-image: radial-gradient(circle at center, black, transparent 82%);
          opacity: 0.32;
          animation: pulse 7s ease-in-out infinite;
        }

        .glow {
          position: absolute;
          inset: auto 0 0 0;
          height: 45%;
          background: linear-gradient(180deg, transparent 0%, rgba(255, 255, 255, 0.08) 100%);
        }

        .card {
          position: relative;
          z-index: 1;
          width: min(920px, 100%);
          padding: 32px;
          border-radius: 28px;
          color: #e5eefb;
          background: rgba(15, 23, 42, 0.58);
          border: 1px solid rgba(255, 255, 255, 0.14);
          backdrop-filter: blur(18px);
          box-shadow:
            0 24px 80px rgba(15, 23, 42, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.14);
          animation: cardFloat 6s ease-in-out infinite;
        }

        .badge {
          display: inline-block;
          margin-bottom: 18px;
          padding: 8px 12px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #bfdbfe;
          background: rgba(59, 130, 246, 0.16);
          border: 1px solid rgba(147, 197, 253, 0.22);
        }

        .logo {
          display: block;
          width: 140px;
          aspect-ratio: 1 / 1;
          object-fit: cover;
          border-radius: 50%;
          border: 2px solid rgba(147, 197, 253, 0.55);
          box-shadow: 0 10px 28px rgba(15, 23, 42, 0.35);
          margin: 0 auto 16px;
        }

        h1 {
          margin: 0 0 10px;
          font-size: clamp(2rem, 5vw, 2.6rem);
          line-height: 1;
        }

        .subtitle {
          margin: 0 0 22px;
          line-height: 1.55;
          color: rgba(226, 232, 240, 0.84);
        }

        .heroStats,
        .singleFieldWrap {
          display: grid;
          gap: 14px;
        }

        .heroStats {
          grid-template-columns: repeat(2, minmax(0, 1fr));
          margin-bottom: 18px;
        }

        .statBox {
          padding: 14px 16px;
          border-radius: 16px;
          background: rgba(15, 23, 42, 0.42);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .statLabel {
          display: block;
          margin-bottom: 6px;
          font-size: 0.78rem;
          color: rgba(191, 219, 254, 0.78);
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .singleFieldWrap {
          grid-template-columns: 1fr;
          margin-bottom: 14px;
        }

        .field {
          display: flex;
          flex-direction: column;
          gap: 8px;
          font-size: 0.95rem;
          color: rgba(226, 232, 240, 0.92);
        }

        .input {
          width: 100%;
          padding: 14px 16px;
          border: 1px solid rgba(255, 255, 255, 0.14);
          border-radius: 14px;
          outline: none;
          font-size: 1rem;
          color: white;
          background: rgba(15, 23, 42, 0.5);
          box-sizing: border-box;
          transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
        }

        .input::placeholder {
          color: rgba(226, 232, 240, 0.45);
        }

        .input:focus {
          border-color: rgba(96, 165, 250, 0.9);
          box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.16);
          transform: translateY(-1px);
        }

        .button {
          width: 100%;
          padding: 14px 16px;
          border: none;
          border-radius: 14px;
          font-size: 1rem;
          font-weight: 700;
          color: white;
          cursor: pointer;
          background: linear-gradient(135deg, #38bdf8 0%, #6366f1 55%, #8b5cf6 100%);
          box-shadow: 0 16px 30px rgba(59, 130, 246, 0.3);
          transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        }

        .button:disabled {
          cursor: not-allowed;
          opacity: 0.7;
        }

        .button:hover {
          transform: translateY(-2px);
          box-shadow: 0 20px 34px rgba(99, 102, 241, 0.38);
          filter: brightness(1.04);
        }

        .button:disabled:hover {
          transform: none;
          box-shadow: 0 16px 30px rgba(59, 130, 246, 0.3);
          filter: none;
        }

        .errorBox {
          margin-top: 16px;
          padding: 14px 16px;
          border-radius: 14px;
          color: #fecaca;
          background: rgba(127, 29, 29, 0.42);
          border: 1px solid rgba(248, 113, 113, 0.22);
        }

        .result {
          margin-top: 18px;
          padding: 16px 18px;
          border-radius: 16px;
          color: #dbeafe;
          background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(37, 99, 235, 0.2));
          border: 1px solid rgba(147, 197, 253, 0.18);
          animation: fadeUp 0.4s ease;
        }

        .resultRow {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 12px;
          font-size: 1.05rem;
        }

        .resultRow strong {
          font-size: 2rem;
        }

        .resultMeta {
          margin-top: 8px;
          color: rgba(219, 234, 254, 0.84);
        }

        @keyframes float {
          0%, 100% {
            transform: translate3d(0, 0, 0) scale(1);
          }
          50% {
            transform: translate3d(22px, -26px, 0) scale(1.08);
          }
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 0.24;
          }
          50% {
            opacity: 0.38;
          }
        }

        @keyframes cardFloat {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-6px);
          }
        }

        @keyframes fadeUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @media (max-width: 640px) {
          .card {
            padding: 24px;
            border-radius: 24px;
          }

          .heroStats,
          .singleFieldWrap {
            grid-template-columns: 1fr;
          }

          .blobOne,
          .blobTwo,
          .blobThree {
            width: 20rem;
            height: 20rem;
          }
        }
      `}</style>
    </main>
  )
}
