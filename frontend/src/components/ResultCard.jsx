import { useState } from "react";
import { submitToBlockchain } from "../api";
import toast from "react-hot-toast";

function SignalRow({ name, value }) {
  const pct = Math.round(value * 100);
  const color = pct > 60 ? "var(--red)" : pct < 40 ? "var(--green)" : "var(--amber)";
  const label = name.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());
  return (
    <div style={{display:"flex", alignItems:"center", gap:12, marginBottom:8}}>
      <span style={{fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--muted)", width:110, flexShrink:0}}>{label}</span>
      <div style={{flex:1, height:6, borderRadius:100, background:"rgba(15,14,23,0.08)"}}>
        <div style={{width:`${pct}%`, height:"100%", borderRadius:100, background:color, transition:"width 0.6s"}}/>
      </div>
      <span style={{fontFamily:"DM Mono,monospace", fontSize:11, color, width:32, textAlign:"right"}}>{pct}%</span>
    </div>
  );
}

function RegionItem({ region, isFake }) {
  const color = isFake ? "var(--red)" : "var(--green)";
  const bg    = isFake ? "var(--red-light)" : "var(--green-light)";
  return (
    <div style={{
      background:"var(--surface)", borderRadius:12, padding:"14px 16px",
      border:"1px solid var(--border)", marginBottom:8,
    }}>
      <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:6}}>
        <span style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:13}}>{region.region}</span>
        <span style={{
          fontFamily:"DM Mono,monospace", fontSize:11,
          padding:"2px 10px", borderRadius:100,
          background:bg, color,
        }}>{region.contribution_pct}%</span>
      </div>
      <div style={{fontFamily:"DM Mono,monospace", fontSize:11, color, marginBottom:4}}>{region.artifact_type}</div>
      <div style={{fontSize:12, color:"var(--muted)", lineHeight:1.6}}>{region.description}</div>
    </div>
  );
}

export default function ResultCard({ result, mediaType }) {
  const [saving,   setSaving]   = useState(false);
  const [saved,    setSaved]    = useState(false);
  const [showXAI,  setShowXAI]  = useState(true);
  const [showSigs, setShowSigs] = useState(false);

  if (!result) return null;

  const detection  = result.detection || result.result || result;
  const xai        = result.xai;
  const isFake     = detection.is_fake;
  const fakeProb   = detection.fake_probability ?? 0.5;
  const pct        = Math.round(fakeProb * 100);
  const realPct    = 100 - pct;
  const fileHash   = detection.file_hash ?? "";
  const signals    = detection.signals || {};
  const votes      = detection.votes || {};
  const modelBreak = result.result?.model_breakdown;

  const accentColor  = isFake ? "var(--red)"   : "var(--green)";
  const accentLight  = isFake ? "var(--red-light)" : "var(--green-light)";
  const accentBorder = isFake ? "rgba(224,62,62,0.2)" : "rgba(26,138,74,0.2)";
  const verdictLabel = isFake ? "FAKE" : "REAL";
  const verdictIcon  = isFake ? "⚠️" : "✅";

  const handleSave = async () => {
    setSaving(true);
    try {
      await submitToBlockchain({
        file_hash: fileHash, is_fake: isFake,
        fake_probability: fakeProb,
        confidence: detection.confidence || "High",
        media_type: mediaType,
        models_used: detection.model || "DeepGuard",
      });
      setSaved(true);
      toast.success("Saved to audit record!");
    } catch(e) {
      toast.error("Save failed: " + e.message);
    } finally { setSaving(false); }
  };

  return (
    <div style={{
      background:"var(--white)", borderRadius:20,
      border:`1px solid ${accentBorder}`,
      boxShadow:`0 4px 24px ${isFake ? "rgba(224,62,62,0.08)" : "rgba(26,138,74,0.08)"}`,
      overflow:"hidden",
    }}>

      {/* ── Verdict banner ───────────────────────────────────────── */}
      <div style={{
        background:accentLight, borderBottom:`1px solid ${accentBorder}`,
        padding:"20px 28px",
        display:"flex", alignItems:"center", justifyContent:"space-between",
      }}>
        <div style={{display:"flex", alignItems:"center", gap:14}}>
          <div style={{
            width:52, height:52, borderRadius:16,
            background:accentColor,
            display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:24,
          }}>{verdictIcon}</div>
          <div>
            <div style={{
              fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:28,
              color:accentColor, lineHeight:1,
            }}>{verdictLabel}</div>
            <div style={{fontSize:13, color:"var(--muted)", marginTop:3}}>
              {isFake ? "AI-generated image detected" : "Authentic media confirmed"}
            </div>
          </div>
        </div>

        {/* Confidence badge */}
        <div style={{textAlign:"center"}}>
          <div style={{
            fontFamily:"Syne,sans-serif", fontWeight:800,
            fontSize:42, color:accentColor, lineHeight:1,
          }}>{isFake ? pct : realPct}%</div>
          <div style={{fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--muted)", marginTop:2}}>
            {isFake ? "fake probability" : "authentic probability"}
          </div>
          <div style={{
            marginTop:6, display:"inline-block",
            fontFamily:"DM Mono,monospace", fontSize:10,
            padding:"2px 10px", borderRadius:100,
            background:accentColor, color:"white",
          }}>{detection.confidence || "High"} confidence</div>
        </div>
      </div>

      <div style={{padding:"24px 28px"}}>

        {/* ── XAI Summary ──────────────────────────────────────────── */}
        {xai?.summary && (
          <div style={{
            display:"flex", gap:10, padding:"12px 16px",
            background:"var(--surface)", borderRadius:12,
            border:"1px solid var(--border)", marginBottom:20,
          }}>
            <span style={{fontSize:16, flexShrink:0}}>🧠</span>
            <p style={{fontSize:13, color:"var(--ink2)", lineHeight:1.65}}>{xai.summary}</p>
          </div>
        )}

        {/* ── Votes row ────────────────────────────────────────────── */}
        {votes.total > 0 && (
          <div style={{
            display:"flex", gap:8, marginBottom:20,
          }}>
            {[
              {label:`${votes.fake || 0} signals → FAKE`, color:"var(--red)",   bg:"var(--red-light)"},
              {label:`${votes.real || 0} signals → REAL`, color:"var(--green)", bg:"var(--green-light)"},
              {label:`${votes.total || 0} total signals`,  color:"var(--muted)", bg:"var(--surface)"},
            ].map((v,i) => (
              <div key={i} style={{
                flex:1, padding:"10px 14px", borderRadius:12,
                background:v.bg, textAlign:"center",
                fontFamily:"Syne,sans-serif", fontWeight:700,
                fontSize:12, color:v.color,
                border:"1px solid var(--border)",
              }}>{v.label}</div>
            ))}
          </div>
        )}

        {/* ── XAI Heatmaps ─────────────────────────────────────────── */}
        {(xai?.heatmap_image || xai?.annotated_image) && (
          <div style={{marginBottom:20}}>
            <button onClick={() => setShowXAI(v=>!v)} style={{
              fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:12,
              color:"var(--violet)", background:"none", border:"none",
              cursor:"pointer", marginBottom:12, display:"flex", alignItems:"center", gap:6,
            }}>
              🔍 {showXAI ? "Hide" : "Show"} AI Explanation
              <span style={{
                fontSize:10, padding:"2px 8px", borderRadius:100,
                background:"var(--violet-light)", color:"var(--violet)",
              }}>{isFake ? "Why it's fake" : "Why it's real"}</span>
            </button>

            {showXAI && (
              <div>
                <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:12}}>
                  {xai.heatmap_image && (
                    <div style={{borderRadius:12, overflow:"hidden", border:"1px solid var(--border)"}}>
                      <div style={{
                        padding:"7px 12px", fontSize:11, fontFamily:"DM Mono,monospace",
                        color:"var(--muted)", background:"var(--surface)",
                        borderBottom:"1px solid var(--border)",
                      }}>
                        {isFake ? "🔴 Suspicious regions" : "🟢 Authentic markers"}
                      </div>
                      <img src={`data:image/png;base64,${xai.heatmap_image}`}
                        alt="Heatmap" style={{width:"100%", maxHeight:160, objectFit:"contain"}} />
                    </div>
                  )}
                  {xai.annotated_image && (
                    <div style={{borderRadius:12, overflow:"hidden", border:"1px solid var(--border)"}}>
                      <div style={{
                        padding:"7px 12px", fontSize:11, fontFamily:"DM Mono,monospace",
                        color:"var(--muted)", background:"var(--surface)",
                        borderBottom:"1px solid var(--border)",
                      }}>
                        {isFake ? "🔴 Artifact annotations" : "🟢 Region annotations"}
                      </div>
                      <img src={`data:image/png;base64,${xai.annotated_image}`}
                        alt="Annotated" style={{width:"100%", maxHeight:160, objectFit:"contain"}} />
                    </div>
                  )}
                </div>

                {/* Region explanations */}
                {xai.regions?.length > 0 && (
                  <div>
                    <div style={{
                      fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:11,
                      color:"var(--muted)", letterSpacing:"0.06em", textTransform:"uppercase",
                      marginBottom:10,
                    }}>
                      {isFake ? "Detected Artifacts" : "Authenticity Markers"}
                    </div>
                    {xai.regions.map((r,i) => <RegionItem key={i} region={r} isFake={isFake}/>)}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Detection signals ─────────────────────────────────────── */}
        {Object.keys(signals).length > 0 && (
          <div style={{marginBottom:20}}>
            <button onClick={() => setShowSigs(v=>!v)} style={{
              fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:12,
              color:"var(--muted)", background:"none", border:"none",
              cursor:"pointer", marginBottom:showSigs?12:0,
            }}>
              📊 {showSigs ? "Hide" : "Show"} Signal Breakdown
            </button>
            {showSigs && (
              <div style={{
                padding:16, background:"var(--surface)",
                borderRadius:12, border:"1px solid var(--border)",
              }}>
                {Object.entries(signals).map(([k,v]) => <SignalRow key={k} name={k} value={v}/>)}
              </div>
            )}
          </div>
        )}

        {/* ── Video model breakdown ─────────────────────────────────── */}
        {modelBreak && (
          <div style={{marginBottom:20}}>
            <div style={{
              fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:11,
              color:"var(--muted)", letterSpacing:"0.06em", textTransform:"uppercase", marginBottom:10,
            }}>Model Breakdown</div>
            <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:10}}>
              {Object.entries(modelBreak).map(([name,info]) => {
                const p = Math.round((info.fake_probability||0.5)*100);
                const c = p > 50 ? "var(--red)" : "var(--green)";
                return (
                  <div key={name} style={{
                    padding:"14px 16px", borderRadius:12,
                    background:"var(--surface)", border:"1px solid var(--border)",
                  }}>
                    <div style={{display:"flex", justifyContent:"space-between", marginBottom:8}}>
                      <span style={{fontSize:12, color:"var(--muted)", textTransform:"capitalize"}}>{name.replace(/_/g," ")}</span>
                      <span style={{fontFamily:"DM Mono,monospace", fontSize:12, fontWeight:700, color:c}}>{p}%</span>
                    </div>
                    <div style={{height:4, borderRadius:100, background:"rgba(15,14,23,0.08)"}}>
                      <div style={{width:`${p}%`, height:"100%", borderRadius:100, background:c}}/>
                    </div>
                    <div style={{fontSize:11, color:"var(--muted)", marginTop:6}}>{info.focus}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Video timeline ────────────────────────────────────────── */}
        {result.result?.timeline?.length > 0 && (
          <div style={{marginBottom:20}}>
            <div style={{
              fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:11,
              color:"var(--muted)", letterSpacing:"0.06em", textTransform:"uppercase", marginBottom:10,
            }}>Frame Timeline</div>
            <div style={{
              display:"flex", gap:3, flexWrap:"wrap",
              padding:12, background:"var(--surface)",
              borderRadius:12, border:"1px solid var(--border)",
            }}>
              {result.result.timeline.map((clip,i) => (
                <div key={i} title={`Frame ${clip.clip_start_frame}: ${Math.round(clip.fake_prob*100)}%`}
                  style={{
                    width:18, height:18, borderRadius:4, cursor:"pointer",
                    background:clip.suspicious
                      ? `rgba(224,62,62,${0.3+clip.fake_prob*0.7})`
                      : `rgba(26,138,74,${0.3+(1-clip.fake_prob)*0.7})`,
                  }}/>
              ))}
            </div>
            <div style={{fontSize:11, color:"var(--muted)", marginTop:6}}>
              Hover each cell · Red = suspicious · Green = authentic
            </div>
          </div>
        )}

        {/* ── Audit trail ───────────────────────────────────────────── */}
        {fileHash && (
          <div style={{
            padding:"16px 18px", borderRadius:14,
            background:"var(--violet-light)",
            border:"1px solid rgba(91,78,232,0.2)",
          }}>
            <div style={{display:"flex", alignItems:"center", gap:8, marginBottom:10}}>
              <span>🔗</span>
              <span style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:13, color:"var(--violet)"}}>
                Blockchain Audit Trail
              </span>
            </div>

            {saved ? (
              <div>
                <div style={{fontSize:13, color:"var(--green)", marginBottom:6}}>✓ Saved to audit record</div>
                <div style={{fontFamily:"DM Mono,monospace", fontSize:10, color:"var(--muted)", wordBreak:"break-all"}}>
                  {fileHash}
                </div>
              </div>
            ) : (
              <div>
                <div style={{fontFamily:"DM Mono,monospace", fontSize:10, color:"var(--muted)", wordBreak:"break-all", marginBottom:10}}>
                  {fileHash}
                </div>
                <button onClick={handleSave} disabled={saving} style={{
                  fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:12,
                  padding:"9px 20px", borderRadius:100,
                  background:"var(--violet)", color:"white",
                  border:"none", cursor:"pointer",
                  opacity: saving ? 0.6 : 1,
                  display:"flex", alignItems:"center", gap:7,
                }}>
                  {saving ? "Saving..." : "Save to Audit Record"}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
