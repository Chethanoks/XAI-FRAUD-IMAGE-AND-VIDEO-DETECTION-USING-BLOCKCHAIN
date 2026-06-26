import { useState } from "react";
import Uploader   from "../components/Uploader";
import ResultCard from "../components/ResultCard";
import { detectImage, detectVideoAndWait } from "../api";
import toast from "react-hot-toast";

const THRESHOLDS = [
  { value:0.55, label:"High Sensitivity",    desc:"Catches more fakes, may flag some real photos" },
  { value:0.65, label:"Balanced (Default)",  desc:"Best balance for most images" },
  { value:0.75, label:"Conservative",        desc:"Fewer false positives on real photos" },
  { value:0.85, label:"Very Conservative",   desc:"Only flags very obvious AI images" },
];

function HowStep({ num, icon, title, desc }) {
  return (
    <div style={{textAlign:"center"}}>
      <div style={{fontSize:32, marginBottom:10}}>{icon}</div>
      <div style={{
        fontFamily:"DM Mono,monospace", fontSize:11,
        letterSpacing:"0.12em", color:"rgba(255,255,255,0.5)",
        marginBottom:6,
      }}>0{num} — {title.toUpperCase()}</div>
      <div style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:15, color:"white", marginBottom:6}}>{title}</div>
      <div style={{fontSize:12, color:"rgba(255,255,255,0.55)", lineHeight:1.65}}>{desc}</div>
    </div>
  );
}

export default function HomePage() {
  const [analyzing,    setAnalyzing]    = useState(false);
  const [result,       setResult]       = useState(null);
  const [mediaType,    setMediaType]    = useState(null);
  const [threshold,    setThreshold]    = useState(0.65);
  const [showSettings, setShowSettings] = useState(false);
  const [progress,     setProgress]     = useState(0);

  const handleAnalyze = async (file) => {
    setAnalyzing(true); setResult(null);
    const isImage = file.type.startsWith("image/");
    setMediaType(isImage ? "image" : "video");
    try {
      const data = isImage
        ? await detectImage(file, threshold)
        : await detectVideoAndWait(file, (d,t) => setProgress(Math.round(d/t*100)), threshold);
      setResult(data);
      const verdict = (data.detection || data.result || data)?.is_fake ? "FAKE" : "REAL";
      toast[verdict==="FAKE"?"error":"success"](`${verdict} — Analysis complete`, {duration:4000});
    } catch(e) {
      toast.error("Analysis failed: " + e.message);
    } finally { setAnalyzing(false); setProgress(0); }
  };

  const reset = () => { setResult(null); setMediaType(null); };

  return (
    <div>
      {/* ── Hero ────────────────────────────────────────────────────────── */}
      {!result && (
        <div style={{maxWidth:1100, margin:"0 auto", padding:"56px 48px 40px", display:"grid", gridTemplateColumns:"1fr 1fr", gap:52, alignItems:"center"}}>
          <div>
            <div style={{
              fontFamily:"DM Mono,monospace", fontSize:11,
              letterSpacing:"0.15em", textTransform:"uppercase",
              color:"var(--violet)", marginBottom:16,
              display:"flex", alignItems:"center", gap:10,
            }}>
              <span style={{width:26, height:1, background:"var(--violet)", display:"inline-block"}}/>
              AI × XAI × Blockchain
            </div>
            <h1 style={{
              fontFamily:"Syne,sans-serif", fontWeight:800,
              fontSize:"clamp(34px,4.5vw,52px)", lineHeight:1.06, marginBottom:18,
            }}>
              Detect <em style={{fontStyle:"italic", color:"var(--violet)"}}>deepfakes</em><br/>
              with forensic precision
            </h1>
            <p style={{fontSize:15, lineHeight:1.75, color:"var(--muted)", marginBottom:32, maxWidth:380}}>
              Upload any image or video. Our multi-model AI analyzes it for deepfake artifacts
              and explains exactly what it found — backed by blockchain audit trail.
            </p>
            <div style={{display:"flex", gap:10, flexWrap:"wrap"}}>
              {[
                {icon:"🧠", label:"EfficientNet-B4 (100k trained)"},
                {icon:"🔍", label:"XAI Explainability"},
                {icon:"⛓️", label:"Blockchain Audit"},
              ].map((f,i) => (
                <div key={i} style={{
                  display:"flex", alignItems:"center", gap:7,
                  fontFamily:"Syne,sans-serif", fontWeight:600, fontSize:12,
                  padding:"8px 14px", borderRadius:100,
                  background:"var(--white)", color:"var(--ink2)",
                  border:"1px solid var(--border)",
                }}>
                  {f.icon} {f.label}
                </div>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div style={{display:"flex", flexDirection:"column", gap:12}}>
            <div style={{
              background:"var(--white)", borderRadius:20, padding:24,
              border:"1px solid var(--border)",
              boxShadow:"0 4px 20px rgba(15,14,23,0.06)",
              display:"flex", alignItems:"center", gap:20,
            }}>
              <div style={{
                width:80, height:80, borderRadius:"50%",
                background:"var(--violet-light)",
                border:"3px solid var(--violet)",
                display:"flex", flexDirection:"column",
                alignItems:"center", justifyContent:"center", flexShrink:0,
              }}>
                <span style={{fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:22, color:"var(--violet)", lineHeight:1}}>95%</span>
                <span style={{fontFamily:"DM Mono,monospace", fontSize:9, color:"var(--muted)"}}>Accuracy</span>
              </div>
              <div>
                <h3 style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:15, marginBottom:4}}>Trained Model Active</h3>
                <div style={{fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--muted)", marginBottom:8}}>EfficientNet-B4 · 100k images</div>
                <div style={{display:"flex", gap:6}}>
                  <span style={{fontFamily:"DM Mono,monospace", fontSize:10, padding:"2px 9px", borderRadius:100, background:"var(--green-light)", color:"var(--green)"}}>✓ Real detection</span>
                  <span style={{fontFamily:"DM Mono,monospace", fontSize:10, padding:"2px 9px", borderRadius:100, background:"var(--red-light)", color:"var(--red)"}}>✓ Fake detection</span>
                </div>
              </div>
            </div>

            <div style={{display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10}}>
              {[
                {v:"100k",  l:"Training images"},
                {v:"XAI",   l:"Explainable AI"},
                {v:"Chain", l:"Audit trail"},
              ].map((s,i) => (
                <div key={i} style={{
                  background:"var(--white)", borderRadius:14, padding:"16px 14px",
                  border:"1px solid var(--border)", textAlign:"center",
                }}>
                  <div style={{fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:20, color:"var(--violet)"}}>{s.v}</div>
                  <div style={{fontSize:11, color:"var(--muted)", marginTop:2}}>{s.l}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Upload / Result section ──────────────────────────────────────── */}
      <div style={{maxWidth:640, margin:"0 auto", padding:"0 24px 48px"}}>

        {/* Video progress */}
        {analyzing && mediaType==="video" && progress>0 && (
          <div style={{marginBottom:16}}>
            <div style={{display:"flex", justifyContent:"space-between", fontSize:12, color:"var(--muted)", marginBottom:6}}>
              <span>Processing video frames...</span><span>{progress}%</span>
            </div>
            <div style={{height:6, borderRadius:100, background:"rgba(15,14,23,0.08)"}}>
              <div style={{width:`${progress}%`, height:"100%", borderRadius:100, background:"var(--violet)", transition:"width 0.3s"}}/>
            </div>
          </div>
        )}

        {!result ? (
          <div>
            {/* Settings toggle */}
            <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12}}>
              <div style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:13, color:"var(--muted)"}}>
                Upload file to analyze
              </div>
              <button onClick={() => setShowSettings(v=>!v)} style={{
                fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--violet)",
                background:"none", border:"none", cursor:"pointer",
                textDecoration:"underline",
              }}>
                {showSettings ? "Hide" : "Sensitivity settings"}
              </button>
            </div>

            {showSettings && (
              <div style={{
                marginBottom:16, padding:18, borderRadius:16,
                background:"var(--white)", border:"1px solid var(--border)",
              }}>
                <div style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:13, marginBottom:12}}>
                  Detection Sensitivity
                </div>
                <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:8}}>
                  {THRESHOLDS.map(t => (
                    <button key={t.value} onClick={() => setThreshold(t.value)} style={{
                      padding:"10px 14px", borderRadius:12, cursor:"pointer",
                      textAlign:"left", transition:"all 0.15s",
                      background: threshold===t.value ? "var(--violet-light)" : "var(--surface)",
                      border: threshold===t.value ? "1.5px solid var(--violet)" : "1px solid var(--border)",
                    }}>
                      <div style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:12, color: threshold===t.value ? "var(--violet)" : "var(--ink)"}}>{t.label}</div>
                      <div style={{fontSize:11, color:"var(--muted)", marginTop:2}}>{t.desc}</div>
                    </button>
                  ))}
                </div>
                <p style={{fontSize:11, color:"var(--muted)", marginTop:10}}>
                  💡 If your real photos show as fake → choose Conservative
                </p>
              </div>
            )}

            <Uploader onAnalyze={handleAnalyze} analyzing={analyzing}/>
          </div>
        ) : (
          <div>
            <ResultCard result={result} mediaType={mediaType}/>
            <button onClick={reset} style={{
              marginTop:12, width:"100%", padding:"12px",
              fontFamily:"Syne,sans-serif", fontWeight:600, fontSize:13,
              background:"var(--white)", color:"var(--muted)",
              border:"1px solid var(--border)", borderRadius:100,
              cursor:"pointer", transition:"all 0.15s",
            }}
            onMouseEnter={e=>{e.target.style.borderColor="var(--violet)";e.target.style.color="var(--violet)"}}
            onMouseLeave={e=>{e.target.style.borderColor="var(--border)";e.target.style.color="var(--muted)"}}
            >
              ← Analyze another file
            </button>
          </div>
        )}
      </div>

      {/* ── How it works ─────────────────────────────────────────────────── */}
      {!result && !analyzing && (
        <div style={{
          background:"var(--ink)", borderRadius:24,
          margin:"0 24px 52px", padding:"48px",
        }}>
          <div style={{maxWidth:1052, margin:"0 auto"}}>
            <div style={{fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:26, color:"white", marginBottom:4}}>
              How DeepGuard works
            </div>
            <div style={{fontSize:13, color:"rgba(255,255,255,0.45)", marginBottom:36}}>
              4 steps from upload to verified result
            </div>
            <div style={{display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:24}}>
              <HowStep num={1} icon="📤" title="Upload"   desc="Drop any image or video. Supports JPG, PNG, MP4, AVI and more up to 200MB."/>
              <HowStep num={2} icon="🧠" title="Analyze"  desc="EfficientNet-B4 model trained on 100k images runs deep forensic analysis."/>
              <HowStep num={3} icon="🔍" title="Explain"  desc="XAI highlights exactly which regions are suspicious and why — for both real and fake."/>
              <HowStep num={4} icon="⛓️" title="Record"   desc="Save the result to an immutable blockchain audit trail on Polygon network."/>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
