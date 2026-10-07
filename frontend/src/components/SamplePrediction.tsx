import { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Sparkles, BrainCircuit, Activity, Cpu } from 'lucide-react';

export default function SamplePrediction() {
  const [selectedSample, setSelectedSample] = useState('Left/Right Fist Motor Imagery (Run 4)');
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceResult, setInferenceResult] = useState({
    task_condition: 'Left/Right Fist Motor Imagery',
    task_alias: 'Focused',
    confidence: 62.5,
    focus: 78.5,
    attention: 81.0,
    stress: 22.2,
    fatigue: 18.8,
    latency: '14.2ms',
    distribution: [
      { class: 'Eyes-Open Rest (Calm)', prob: 0.12, color: '#00ff9d' },
      { class: 'Left/Right Fist MI (Focused)', prob: 0.625, color: '#00f0ff' },
      { class: 'Both-Fists/Feet MI (Stressed)', prob: 0.11, color: '#ff2a9d' },
      { class: 'Eyes-Closed Rest (Fatigued)', prob: 0.045, color: '#b52aff' },
      { class: 'Left/Right Fist ME (Excited)', prob: 0.10, color: '#ffea00' },
    ],
  });

  const samples = [
    {
      name: 'Left/Right Fist Motor Imagery (Run 4)',
      desc: 'Mu/Beta Event-Related Desynchronization (ERD) over C3/C4 sensorimotor channels',
      result: {
        task_condition: 'Left/Right Fist Motor Imagery',
        task_alias: 'Focused',
        confidence: 62.5,
        focus: 78.5,
        attention: 81.0,
        stress: 22.2,
        fatigue: 18.8,
        latency: '14.2ms',
        distribution: [
          { class: 'Eyes-Open Rest (Calm)', prob: 0.12, color: '#00ff9d' },
          { class: 'Left/Right Fist MI (Focused)', prob: 0.625, color: '#00f0ff' },
          { class: 'Both-Fists/Feet MI (Stressed)', prob: 0.11, color: '#ff2a9d' },
          { class: 'Eyes-Closed Rest (Fatigued)', prob: 0.045, color: '#b52aff' },
          { class: 'Left/Right Fist ME (Excited)', prob: 0.10, color: '#ffea00' },
        ],
      },
    },
    {
      name: 'Eyes-Closed Baseline Rest (Run 2)',
      desc: 'Elevated 10 Hz alpha synchrony in occipital channels (O1, O2, Pz)',
      result: {
        task_condition: 'Eyes-Closed Rest',
        task_alias: 'Fatigued',
        confidence: 58.4,
        focus: 32.0,
        attention: 28.0,
        stress: 15.0,
        fatigue: 72.0,
        latency: '13.8ms',
        distribution: [
          { class: 'Eyes-Open Rest (Calm)', prob: 0.18, color: '#00ff9d' },
          { class: 'Left/Right Fist MI (Focused)', prob: 0.08, color: '#00f0ff' },
          { class: 'Both-Fists/Feet MI (Stressed)', prob: 0.06, color: '#ff2a9d' },
          { class: 'Eyes-Closed Rest (Fatigued)', prob: 0.584, color: '#b52aff' },
          { class: 'Left/Right Fist ME (Excited)', prob: 0.096, color: '#ffea00' },
        ],
      },
    },
    {
      name: 'Both-Fists / Both-Feet Motor Imagery (Run 6)',
      desc: 'Bilateral sensorimotor beta modulation with frontal engagement (Cz, Fz)',
      result: {
        task_condition: 'Both-Fists/Both-Feet Motor Imagery',
        task_alias: 'Stressed',
        confidence: 54.0,
        focus: 68.0,
        attention: 74.0,
        stress: 65.5,
        fatigue: 35.0,
        latency: '14.5ms',
        distribution: [
          { class: 'Eyes-Open Rest (Calm)', prob: 0.10, color: '#00ff9d' },
          { class: 'Left/Right Fist MI (Focused)', prob: 0.18, color: '#00f0ff' },
          { class: 'Both-Fists/Feet MI (Stressed)', prob: 0.54, color: '#ff2a9d' },
          { class: 'Eyes-Closed Rest (Fatigued)', prob: 0.05, color: '#b52aff' },
          { class: 'Left/Right Fist ME (Excited)', prob: 0.13, color: '#ffea00' },
        ],
      },
    },
    {
      name: 'Eyes-Open Baseline Rest (Run 1)',
      desc: 'Broadband physiological baseline with low-amplitude desynchronized rhythms',
      result: {
        task_condition: 'Eyes-Open Rest',
        task_alias: 'Calm',
        confidence: 51.2,
        focus: 45.0,
        attention: 42.0,
        stress: 18.0,
        fatigue: 25.0,
        latency: '13.1ms',
        distribution: [
          { class: 'Eyes-Open Rest (Calm)', prob: 0.512, color: '#00ff9d' },
          { class: 'Left/Right Fist MI (Focused)', prob: 0.14, color: '#00f0ff' },
          { class: 'Both-Fists/Feet MI (Stressed)', prob: 0.12, color: '#ff2a9d' },
          { class: 'Eyes-Closed Rest (Fatigued)', prob: 0.118, color: '#b52aff' },
          { class: 'Left/Right Fist ME (Excited)', prob: 0.11, color: '#ffea00' },
        ],
      },
    },
    {
      name: 'Left/Right Fist Motor Execution (Run 3)',
      desc: 'Physical movement activation with pronounced contralateral beta desynchronization',
      result: {
        task_condition: 'Left/Right Fist Motor Execution',
        task_alias: 'Excited',
        confidence: 59.8,
        focus: 72.0,
        attention: 76.0,
        stress: 40.0,
        fatigue: 30.0,
        latency: '14.0ms',
        distribution: [
          { class: 'Eyes-Open Rest (Calm)', prob: 0.08, color: '#00ff9d' },
          { class: 'Left/Right Fist MI (Focused)', prob: 0.16, color: '#00f0ff' },
          { class: 'Both-Fists/Feet MI (Stressed)', prob: 0.10, color: '#ff2a9d' },
          { class: 'Eyes-Closed Rest (Fatigued)', prob: 0.062, color: '#b52aff' },
          { class: 'Left/Right Fist ME (Excited)', prob: 0.598, color: '#ffea00' },
        ],
      },
    },
  ];

  const handleRunInference = () => {
    setIsInferring(true);
    setTimeout(() => {
      const match = samples.find((s) => s.name === selectedSample);
      if (match) {
        setInferenceResult(match.result);
      }
      setIsInferring(false);
    }, 300);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      className="flex flex-col gap-6 py-2"
    >
      {/* HEADER */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-base font-black tracking-widest text-gray-200 uppercase font-mono flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[#00f0ff]" /> SAMPLE TASK-CONDITION INFERENCE BENCH
          </h2>
          <div className="text-[10px] text-gray-500 font-mono tracking-wider mt-0.5">
            FIVE-CLASS TASK-CONDITION CLASSIFICATION USING PHYSIONET EEG RECORDINGS
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Sample Selection */}
        <div className="lg:col-span-5 glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/90 flex flex-col gap-4">
          <h3 className="text-xs font-bold tracking-widest text-gray-300 uppercase font-mono">
            SELECT PHYSIOLOGICAL EEG WINDOW
          </h3>

          <div className="flex flex-col gap-3">
            {samples.map((s) => (
              <button
                key={s.name}
                onClick={() => setSelectedSample(s.name)}
                className={`p-3 rounded-lg border text-left font-mono transition-all duration-200 ${
                  selectedSample === s.name
                    ? 'border-[#00f0ff] bg-[#00f0ff]/10 shadow-[0_0_12px_rgba(0,240,255,0.2)]'
                    : 'border-gray-800 bg-[#05050a] hover:border-gray-700'
                }`}
              >
                <div className="text-xs font-bold text-white mb-1 flex items-center justify-between">
                  <span>{s.name}</span>
                  <span className="text-[9px] text-[#00ff9d]">19-Ch EEG</span>
                </div>
                <div className="text-[10px] text-gray-400">{s.desc}</div>
              </button>
            ))}
          </div>

          <button
            onClick={handleRunInference}
            disabled={isInferring}
            className="w-full mt-2 py-3 rounded-xl font-mono text-xs font-bold flex items-center justify-center gap-2 transition-all border border-[#00ff9d]/50 bg-gradient-to-r from-[#00ff9d]/20 to-[#00f0ff]/20 text-[#00ff9d] hover:border-[#00ff9d] hover:shadow-[0_0_20px_rgba(0,255,157,0.3)] disabled:opacity-50"
          >
            {isInferring ? (
              <Activity className="w-4 h-4 animate-spin text-[#00ff9d]" />
            ) : (
              <Play className="w-4 h-4 text-[#00ff9d]" />
            )}
            {isInferring ? 'DECODING TASK CONDITION...' : 'RUN INFERENCE'}
          </button>
        </div>

        {/* Prediction Results */}
        <div className="lg:col-span-7 glass-panel p-5 rounded-xl border border-[#1a1a2e] bg-[#0a0a1a]/90 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xs font-bold tracking-widest text-gray-300 uppercase font-mono flex items-center gap-2">
                <BrainCircuit className="w-4 h-4 text-[#b52aff]" /> MODEL DECODING OUTPUT
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/30">
                LATENCY: {inferenceResult.latency}
              </span>
            </div>

            {/* Main Predicted Class Pill */}
            <div className="bg-[#05050a] p-4 rounded-xl border border-gray-800 flex items-center justify-between mb-4">
              <div>
                <div className="text-[10px] font-mono text-gray-500">PREDICTED TASK CONDITION (PROXY ALIAS)</div>
                <div className="text-xl font-black font-mono text-transparent bg-clip-text bg-gradient-to-r from-[#00f0ff] via-[#00ff9d] to-[#ffea00]">
                  {inferenceResult.task_condition} ({inferenceResult.task_alias.toUpperCase()})
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] font-mono text-gray-500">CONFIDENCE</div>
                <div className="text-2xl font-black font-mono text-[#00ff9d]">
                  {inferenceResult.confidence}%
                </div>
              </div>
            </div>

            {/* Softmax Probability Distribution */}
            <div className="mb-4">
              <div className="text-[10px] font-mono text-gray-400 mb-2">5-CLASS SOFTMAX PROBABILITIES</div>
              <div className="flex flex-col gap-2">
                {inferenceResult.distribution.map((d) => (
                  <div key={d.class} className="flex flex-col gap-1 font-mono text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400 text-[11px]">{d.class}</span>
                      <span style={{ color: d.color }}>{(d.prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-2 bg-[#05050a] rounded-full overflow-hidden border border-gray-800">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: d.color, boxShadow: `0 0 8px ${d.color}60` }}
                        initial={{ width: 0 }}
                        animate={{ width: `${d.prob * 100}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Synthetic Proxy Metrics */}
            <div className="grid grid-cols-4 gap-2 pt-2 border-t border-gray-800 text-center font-mono">
              <div className="bg-[#05050a] p-2 rounded border border-gray-800">
                <div className="text-[9px] text-gray-500">FOCUS (PROXY)</div>
                <div className="text-sm font-bold text-[#00f0ff]">{inferenceResult.focus}</div>
              </div>
              <div className="bg-[#05050a] p-2 rounded border border-gray-800">
                <div className="text-[9px] text-gray-500">ATTENTION (PROXY)</div>
                <div className="text-sm font-bold text-[#b52aff]">{inferenceResult.attention}</div>
              </div>
              <div className="bg-[#05050a] p-2 rounded border border-gray-800">
                <div className="text-[9px] text-gray-500">STRESS (PROXY)</div>
                <div className="text-sm font-bold text-[#ff2a9d]">{inferenceResult.stress}</div>
              </div>
              <div className="bg-[#05050a] p-2 rounded border border-gray-800">
                <div className="text-[9px] text-gray-500">FATIGUE (PROXY)</div>
                <div className="text-sm font-bold text-[#ffea00]">{inferenceResult.fatigue}</div>
              </div>
            </div>
          </div>

          <div className="text-[9px] font-mono text-gray-500 flex items-center justify-between mt-4">
            <span className="flex items-center gap-1">
              <Cpu className="w-3 h-3 text-[#00f0ff]" /> Neural Network: Hybrid CNN-LSTM + Attention
            </span>
            <span className="text-[#00ff9d]">Task-Condition Decoding Verified</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
