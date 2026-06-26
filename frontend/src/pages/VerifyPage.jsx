import { useState } from "react";
import { verifyByHash } from "../api";
import toast from "react-hot-toast";

function StatBox({ label, value, color }) {
  return (
    <div style={{
      flex:1, padding:"16px 14px", borderRadius:14, textAlign:"center",
      background:"var(--white)", border:"1px solid var(--border)",
    }}>
      <div style={{fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:22, color:color||"var(--ink)"}}>{value}</div>
      <div style={{fontSize:11, color:"var(--muted)", marginTop:3}}>{label}</div>
    </div>
  );
}

function HistoryRow({ record, index }) {
  const isFake = record.verdict === "FAKE";
  const date   = new Date(record.timestamp * 1000).toLocaleString();
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:12,
      padding:"12px 16px", borderRadius:12,
      background:"var(--surface)", border:"1px solid var(--border)",
      marginBottom:8,
    }}>
      <span style={{fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--muted)", width:20}}>#{index+1}</span>
      <span style={{
        fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:13,
        color: isFake ? "var(--red)" : "var(--green)", width:40,
      }}>{record.verdict}</span>
      <span style={{fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--muted)"}}>
        {Math.round(record.fake_probability*100)}% fake
      </span>
      <span style={{fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--muted)"}}>
        {record.confidence}
      </span>
      <span style={{fontFamily:"DM Mono,monospace", fontSize:11, color:"var(--muted)", marginLeft:"auto"}}>
        {date}
      </span>
    </div>
  );
}

export default function VerifyPage() {
  const [hash,    setHash]    = useState("");
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  const handleVerify = async () => {
    const h = hash.trim();
    if (h.length !== 64) { toast.error("Hash must be 64 hex characters (SHA-256)"); return; }
    setLoading(true); setResult(null);
    try {
      setResult(await verifyByHash(h));
    } catch(e) {
      toast.error("Lookup failed: " + e.message);
    } finally { setLoading(false); }
  };

  return (
    <div style={{maxWidth:680, margin:"0 auto", padding:"52px 24px"}}>

      {/* Header */}
      <div style={{textAlign:"center", marginBottom:40}}>
        <div style={{
          fontFamily:"DM Mono,monospace", fontSize:11,
          letterSpacing:"0.15em", textTransform:"uppercase",
          color:"var(--violet)", marginBottom:14,
          display:"flex", alignItems:"center", justifyContent:"center", gap:10,
        }}>
          <span style={{width:26, height:1, background:"var(--violet)", display:"inline-block"}}/>
          Public Verification Portal
        </div>
        <h1 style={{fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:38, marginBottom:12}}>
          Verify Any File
        </h1>
        <p style={{fontSize:14, color:"var(--muted)", lineHeight:1.7, maxWidth:420, margin:"0 auto"}}>
          Paste the SHA-256 hash of any file to look up its detection history in the audit record.
        </p>
      </div>

      {/* Search */}
      <div style={{display:"flex", gap:10, marginBottom:32}}>
        <div style={{flex:1, position:"relative"}}>
          <span style={{
            position:"absolute", left:14, top:"50%", transform:"translateY(-50%)",
            fontFamily:"DM Mono,monospace", fontSize:14, color:"var(--muted)",
          }}>#</span>
          <input
            value={hash}
            onChange={e => setHash(e.target.value)}
            onKeyDown={e => e.key==="Enter" && handleVerify()}
            placeholder="SHA-256 hash (64 hex characters)..."
            style={{
              width:"100%", padding:"12px 16px 12px 32px",
              fontFamily:"DM Mono,monospace", fontSize:12,
              border:"1.5px solid var(--border2)", borderRadius:100,
              background:"var(--white)", color:"var(--ink)",
              outline:"none", transition:"border-color 0.2s",
            }}
            onFocus={e => e.target.style.borderColor="var(--violet)"}
            onBlur={e  => e.target.style.borderColor="var(--border2)"}
          />
        </div>
        <button onClick={handleVerify} disabled={loading} style={{
          fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:13,
          padding:"12px 24px", borderRadius:100,
          background:"var(--violet)", color:"white",
          border:"none", cursor:"pointer", opacity: loading ? 0.6 : 1,
          display:"flex", alignItems:"center", gap:8,
          transition:"all 0.18s",
        }}>
          {loading ? (
            <>
              <span style={{width:14,height:14,border:"2px solid rgba(255,255,255,0.3)",borderTopColor:"white",borderRadius:"50%",display:"inline-block",animation:"spin 0.8s linear infinite"}}/>
              Searching...
            </>
          ) : "🔍 Verify"}
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </button>
      </div>

      {/* Result */}
      {result && (
        <div style={{
          background:"var(--white)", borderRadius:20,
          border:"1px solid var(--border)",
          boxShadow:"0 4px 20px rgba(15,14,23,0.06)",
          overflow:"hidden",
        }}>
          {!result.analyzed ? (
            <div style={{padding:"52px 24px", textAlign:"center"}}>
              <div style={{fontSize:40, marginBottom:14}}>🔎</div>
              <div style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:16, marginBottom:8}}>
                No records found
              </div>
              <div style={{fontSize:13, color:"var(--muted)"}}>
                This file hasn't been analyzed yet. Upload it on the home page.
              </div>
            </div>
          ) : (
            <div>
              {/* Verdict banner */}
              <div style={{
                padding:"20px 28px",
                background: result.latest_verdict==="FAKE" ? "var(--red-light)" : "var(--green-light)",
                borderBottom: `1px solid ${result.latest_verdict==="FAKE" ? "rgba(224,62,62,0.2)" : "rgba(26,138,74,0.2)"}`,
                display:"flex", alignItems:"center", gap:16,
              }}>
                <span style={{fontSize:32}}>{result.latest_verdict==="FAKE" ? "⚠️" : "✅"}</span>
                <div>
                  <div style={{
                    fontFamily:"Syne,sans-serif", fontWeight:800, fontSize:28,
                    color: result.latest_verdict==="FAKE" ? "var(--red)" : "var(--green)",
                    lineHeight:1,
                  }}>{result.latest_verdict}</div>
                  <div style={{fontSize:13, color:"var(--muted)", marginTop:3}}>Latest detection verdict</div>
                </div>
              </div>

              <div style={{padding:"24px 28px"}}>
                {/* Stats */}
                <div style={{display:"flex", gap:10, marginBottom:24}}>
                  <StatBox label="Total Analyses"  value={result.total_analyses} />
                  <StatBox label="Fake Verdicts"   value={result.fake_verdicts}  color="var(--red)" />
                  <StatBox label="Real Verdicts"   value={result.real_verdicts}  color="var(--green)" />
                  <StatBox label="Avg Fake Prob"   value={`${result.avg_fake_probability}%`} />
                </div>

                {/* Timestamp */}
                {result.latest_timestamp && (
                  <div style={{
                    display:"flex", alignItems:"center", gap:8,
                    fontFamily:"DM Mono,monospace", fontSize:12,
                    color:"var(--muted)", marginBottom:20,
                  }}>
                    🕐 Last analyzed: {new Date(result.latest_timestamp*1000).toLocaleString()}
                  </div>
                )}

                {/* History */}
                {result.detection_history?.length > 0 && (
                  <div>
                    <div style={{
                      fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:11,
                      color:"var(--muted)", letterSpacing:"0.06em",
                      textTransform:"uppercase", marginBottom:10,
                    }}>Detection History</div>
                    {result.detection_history.map((rec,i) => <HistoryRow key={i} record={rec} index={i}/>)}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* How it works */}
      <div style={{
        marginTop:40, padding:"22px 24px",
        background:"var(--white)", borderRadius:20,
        border:"1px solid var(--border)",
      }}>
        <div style={{fontFamily:"Syne,sans-serif", fontWeight:700, fontSize:14, marginBottom:14}}>
          How verification works
        </div>
        <div style={{display:"flex", flexDirection:"column", gap:10}}>
          {[
            ["📤","Upload your file on the home page — our AI analyzes it for deepfake artifacts."],
            ["🔐","The result is saved with the file's SHA-256 hash as an immutable audit record."],
            ["🔍","Anyone can paste the hash here to see the full detection history."],
            ["🛡️","No file content is ever stored — only the hash and detection verdict."],
          ].map(([icon,text],i) => (
            <div key={i} style={{display:"flex", gap:12, alignItems:"flex-start"}}>
              <span style={{fontSize:16, flexShrink:0}}>{icon}</span>
              <p style={{fontSize:13, color:"var(--muted)", lineHeight:1.65}}>{text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
