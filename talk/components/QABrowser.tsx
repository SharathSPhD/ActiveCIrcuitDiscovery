'use client';

import { useState } from 'react';
import { QAS, QA } from '../data/qa';

const TAGS = ['All', 'EFE theory', 'Mech interp', 'Statistics', 'Scaling & safety', 'Rapid fire'] as const;

export default function QABrowser() {
  const [tag, setTag] = useState<(typeof TAGS)[number]>('All');
  const [openAll, setOpenAll] = useState(false);
  const list = QAS.filter((q) => tag === 'All' || q.tag === tag);

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '1.2rem 0 1.6rem', alignItems: 'center' }}>
        {TAGS.map((t) => (
          <button
            key={t}
            onClick={() => setTag(t)}
            style={{
              fontFamily: 'var(--grotesk)', fontSize: '.78rem', fontWeight: 600,
              padding: '6px 14px', borderRadius: 999, cursor: 'pointer',
              border: `1px solid ${tag === t ? 'var(--teal-bright)' : 'var(--navy-hairline)'}`,
              background: tag === t ? 'rgba(79,216,206,.14)' : 'transparent',
              color: tag === t ? 'var(--teal-bright)' : 'var(--cream-soft)',
            }}
          >
            {t}
            <span style={{ opacity: 0.6, marginLeft: 6 }}>
              {t === 'All' ? QAS.length : QAS.filter((q) => q.tag === t).length}
            </span>
          </button>
        ))}
        <button
          onClick={() => setOpenAll((v) => !v)}
          style={{ marginLeft: 'auto', fontFamily: 'var(--grotesk)', fontSize: '.74rem', padding: '6px 12px', borderRadius: 999, cursor: 'pointer', border: '1px solid var(--navy-hairline)', background: 'transparent', color: 'var(--cream-soft)' }}
        >
          {openAll ? 'collapse all' : 'expand all'}
        </button>
      </div>
      {list.map((qa: QA, i) => (
        <details key={qa.id} className="qa-card" open={openAll}>
          <summary>
            <span style={{ fontFamily: 'var(--mono)', fontSize: '.7rem', opacity: 0.5, minWidth: 26 }}>
              {String(i + 1).padStart(2, '0')}
            </span>
            <span className="qa-q">{qa.q}</span>
            <span className="qa-tag">{qa.tag}</span>
          </summary>
          <div className="qa-body">
            {qa.concede && (
              <>
                <div className="qa-label concede">Concede first</div>
                <div>{qa.concede}</div>
              </>
            )}
            <div className="qa-label reply">{qa.concede ? 'Then the reply' : 'Answer'}</div>
            <div>{qa.reply}</div>
          </div>
        </details>
      ))}
    </div>
  );
}
