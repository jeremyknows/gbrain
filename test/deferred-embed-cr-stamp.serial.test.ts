/**
 * Regression: a noEmbed import followed by an embed pass ends where an inline
 * import does (Atlas fork, 2026-10-18, cf-ops-258).
 *
 * The autopilot cycle syncs with noEmbed:true and embeds in a separate phase.
 * import-file.ts skips the contextual-retrieval stamp on noEmbed and promises
 * the next embed pass will set it; before this fix no embed pass did. New
 * pages landed with NULL corpus_generation and vectors without the title
 * wrapper. These tests drive both deferred paths on PGLite:
 *   - runEmbedCore({ stale: true }), the cycle's embed phase and `embed --stale`
 *   - embedStaleForSource, the embed-backfill Minion
 * and check the stamp and the embedder input against an inline import.
 */
import { describe, test, expect, mock, beforeAll, afterAll, beforeEach } from 'bun:test';

// Capture every string sent to the embedder.
let embeddedTexts: string[] = [];
// Keep the module's other exports; only the embedder call is faked.
const realEmbedding = await import('../src/core/embedding.ts');
mock.module('../src/core/embedding.ts', () => ({
  ...realEmbedding,
  embedBatch: async (texts: string[]) => {
    embeddedTexts.push(...texts);
    return texts.map(() => {
      const v = new Float32Array(1536);
      v[0] = 1;
      return v;
    });
  },
  currentEmbeddingSignature: () => 'test:model:1536',
}));

const { PGLiteEngine } = await import('../src/core/pglite-engine.ts');
const { resetPgliteState } = await import('./helpers/reset-pglite.ts');
const { importFromContent } = await import('../src/core/import-file.ts');
const { runEmbedCore } = await import('../src/commands/embed.ts');
const { embedStaleForSource } = await import('../src/core/embed-stale.ts');
const { computeCorpusGeneration } = await import('../src/core/contextual-retrieval-service.ts');
const { buildContextualPrefix } = await import('../src/core/embedding-context.ts');
const { pageFullyEmbeddedThisPass, prunePassCoverage } = await import('../src/core/inline-cr-shape.ts');
const { __setEmbedTransportForTests } = await import('../src/core/ai/gateway.ts');
// runEmbedCore preflights embedding credentials through the gateway seam.
__setEmbedTransportForTests(async () => ({ embeddings: [], usage: { tokens: 0 } } as any));

type Engine = InstanceType<typeof PGLiteEngine>;
let engine: Engine;

beforeAll(async () => {
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();
}, 30000);

afterAll(async () => {
  __setEmbedTransportForTests(null);
  await engine.disconnect();
});

beforeEach(async () => {
  await resetPgliteState(engine);
  embeddedTexts = [];
});

const TITLE = 'Deferred Embed Probe';
const page = (body: string) => `---\ntitle: ${TITLE}\ntype: note\n---\n\n${body}\n`;
const BODY = 'The deferred embed pass must stamp this page and wrap its chunks with the title.';
const PREFIX = buildContextualPrefix(TITLE, null) as string;
const TITLE_GEN = computeCorpusGeneration({
  crMode: 'title',
  haikuModel: 'anthropic:claude-haiku-4-5-20251001',
});

async function crState(slug: string) {
  const rows = await engine.executeRaw<{ mode: string | null; gen: string | null }>(
    `SELECT contextual_retrieval_mode AS mode, corpus_generation AS gen FROM pages WHERE slug = $1`,
    [slug],
  );
  return rows[0];
}

async function staleChunkCount(slug: string) {
  const rows = await engine.executeRaw<{ n: number }>(
    `SELECT count(*)::int AS n FROM content_chunks c JOIN pages p ON p.id = c.page_id
      WHERE p.slug = $1 AND c.embedding IS NULL`,
    [slug],
  );
  return rows[0].n;
}

describe('deferred embed stamps contextual retrieval like an inline import', () => {
  test('inline import: title wrapper and title-tier stamp (the reference)', async () => {
    await importFromContent(engine, 'probe/inline', page(BODY), {});
    const st = await crState('probe/inline');
    expect(st.mode).toBe('title');
    expect(st.gen).toBe(TITLE_GEN);
    expect(embeddedTexts.length).toBeGreaterThan(0);
    for (const t of embeddedTexts) expect(t.startsWith(PREFIX)).toBe(true);
  });

  test('cycle path: noEmbed import then runEmbedCore({stale}) stamps and wraps', async () => {
    await importFromContent(engine, 'probe/cycle', page(BODY), { noEmbed: true });
    expect((await crState('probe/cycle')).gen).toBeNull();
    expect(await staleChunkCount('probe/cycle')).toBeGreaterThan(0);
    expect(embeddedTexts).toEqual([]);

    await runEmbedCore(engine, { stale: true });

    expect(await staleChunkCount('probe/cycle')).toBe(0);
    const st = await crState('probe/cycle');
    expect(st.mode).toBe('title');
    expect(st.gen).toBe(TITLE_GEN);
    expect(embeddedTexts.length).toBeGreaterThan(0);
    for (const t of embeddedTexts) expect(t.startsWith(PREFIX)).toBe(true);
  });

  test('cycle path sends the embedder exactly what an inline import sends', async () => {
    await importFromContent(engine, 'probe/a', page(BODY), {});
    const inline = [...embeddedTexts];
    embeddedTexts = [];
    await importFromContent(engine, 'probe/b', page(BODY), { noEmbed: true });
    await runEmbedCore(engine, { stale: true });
    expect(embeddedTexts).toEqual(inline);
    expect(await crState('probe/b')).toEqual(await crState('probe/a'));
  });

  test('Minion path: noEmbed import then embedStaleForSource stamps and wraps', async () => {
    await importFromContent(engine, 'probe/minion', page(BODY), { noEmbed: true });
    const seen: string[] = [];
    const result = await embedStaleForSource(engine, 'default', {
      embedFn: async (texts) => {
        seen.push(...texts);
        return texts.map(() => {
          const v = new Float32Array(1536);
          v[0] = 1;
          return v;
        });
      },
    });
    expect(result.done).toBe(true);
    const st = await crState('probe/minion');
    expect(st.mode).toBe('title');
    expect(st.gen).toBe(TITLE_GEN);
    expect(seen.length).toBeGreaterThan(0);
    for (const t of seen) expect(t.startsWith(PREFIX)).toBe(true);
  });

  test('a partially stale page keeps its existing stamp', async () => {
    await importFromContent(engine, 'probe/partial', page(`${BODY}\n\n## Two\n\n${Array.from({ length: 600 }, (_, i) => `Sentence ${i} about the probe.`).join(' ')}`), {});
    const before = await crState('probe/partial');
    expect(before.gen).toBe(TITLE_GEN);
    // Mark the page as stamped by some other generation, then NULL one chunk.
    await engine.executeRaw(`UPDATE pages SET corpus_generation = 'older-gen' WHERE slug = $1`, ['probe/partial']);
    const chunks = await engine.executeRaw<{ n: number }>(
      `SELECT count(*)::int AS n FROM content_chunks c JOIN pages p ON p.id = c.page_id WHERE p.slug = $1`,
      ['probe/partial'],
    );
    expect(chunks[0].n).toBeGreaterThan(1);
    await engine.executeRaw(
      `UPDATE content_chunks SET embedding = NULL, embedded_at = NULL
        WHERE chunk_index = 0 AND page_id = (SELECT id FROM pages WHERE slug = $1)`,
      ['probe/partial'],
    );
    embeddedTexts = [];
    await runEmbedCore(engine, { stale: true });
    expect(await staleChunkCount('probe/partial')).toBe(0);
    expect((await crState('probe/partial')).gen).toBe('older-gen');
    expect(embeddedTexts.length).toBe(1);
    expect(embeddedTexts[0].startsWith(PREFIX)).toBe(true);
  });

  test('kill switch: raw text, stamped none with NULL generation, as inline does', async () => {
    await engine.setConfig('search.contextual_retrieval_disabled', 'true');
    try {
      await importFromContent(engine, 'probe/off-inline', page(BODY), {});
      const inline = [...embeddedTexts];
      embeddedTexts = [];
      await importFromContent(engine, 'probe/off-cycle', page(BODY), { noEmbed: true });
      await runEmbedCore(engine, { stale: true });
      expect(embeddedTexts).toEqual(inline);
      for (const t of embeddedTexts) expect(t.startsWith('<context>')).toBe(false);
      expect(await crState('probe/off-cycle')).toEqual({ mode: 'none', gen: null });
      expect(await crState('probe/off-inline')).toEqual({ mode: 'none', gen: null });
    } finally {
      await engine.setConfig('search.contextual_retrieval_disabled', 'false');
    }
  });

  // The freshness watchdog said NULL-generation pages are "invisible to search".
  // No search SQL reads corpus_generation (it keys query-cache invalidation),
  // so pin the actual behaviour: such a page is still found.
  test('a NULL-generation page is still returned by keyword and vector search', async () => {
    await importFromContent(engine, 'probe/nullgen', page('Zanzibarquokka deferred visibility marker.'), {});
    await engine.executeRaw(
      `UPDATE pages SET corpus_generation = NULL, contextual_retrieval_mode = NULL WHERE slug = $1`,
      ['probe/nullgen'],
    );
    expect(await crState('probe/nullgen')).toEqual({ mode: null, gen: null });
    const kw = await engine.searchKeyword('Zanzibarquokka', { limit: 5 });
    expect(kw.map((r) => r.slug)).toContain('probe/nullgen');
    const v = new Float32Array(1536);
    v[0] = 1;
    const vec = await engine.searchVector(v, { limit: 5 });
    expect(vec.map((r) => r.slug)).toContain('probe/nullgen');
  });

  // Review finding (Codex, 2026-10-18): keyset batches can split one page's
  // stale chunks; the stamp must still land once the whole page is embedded.
  test('a page whose stale chunks straddle batches is still stamped (cycle and Minion)', async () => {
    const long = `${BODY}\n\n## Two\n\n${Array.from({ length: 900 }, (_, i) => `Sentence ${i} about the probe.`).join(' ')}`;
    await importFromContent(engine, 'probe/straddle-cycle', page(long), { noEmbed: true });
    await importFromContent(engine, 'probe/straddle-minion', page(long), { noEmbed: true, sourceId: 'default' });
    const n = await staleChunkCount('probe/straddle-cycle');
    expect(n).toBeGreaterThan(2);
    await runEmbedCore(engine, { stale: true, batchSize: 2 });
    expect(await staleChunkCount('probe/straddle-cycle')).toBe(0);
    expect((await crState('probe/straddle-cycle')).gen).toBe(TITLE_GEN);
    expect((await crState('probe/straddle-minion')).gen).toBe(TITLE_GEN);

    await importFromContent(engine, 'probe/straddle-only-minion', page(`${long} Extra.`), { noEmbed: true });
    const res = await embedStaleForSource(engine, 'default', {
      batchSize: 2,
      embedFn: async (texts) => texts.map(() => { const v = new Float32Array(1536); v[0] = 1; return v; }),
    });
    expect(res.done).toBe(true);
    expect((await crState('probe/straddle-only-minion')).gen).toBe(TITLE_GEN);
  });

  // Review round 2 (Codex): the embedding signature had the same batch-split
  // bug; both stamps now share the whole-pass rule.
  test('the embedding signature is stamped on a page that straddles batches', async () => {
    const long = `${BODY}\n\n## Two\n\n${Array.from({ length: 900 }, (_, i) => `Sentence ${i} about the probe.`).join(' ')}`;
    await importFromContent(engine, 'probe/sig-straddle', page(long), { noEmbed: true });
    expect(await staleChunkCount('probe/sig-straddle')).toBeGreaterThan(2);
    await embedStaleForSource(engine, 'default', {
      batchSize: 2,
      embeddingSignature: 'test:sig:1',
      embedFn: async (texts) => texts.map(() => { const v = new Float32Array(1536); v[0] = 1; return v; }),
    });
    const rows = await engine.executeRaw<{ sig: string | null }>(
      `SELECT embedding_signature AS sig FROM pages WHERE slug = $1`,
      ['probe/sig-straddle'],
    );
    expect(rows[0].sig).toBe('test:sig:1');
    expect((await crState('probe/sig-straddle')).gen).toBe(TITLE_GEN);
  });
});

describe('pageFullyEmbeddedThisPass', () => {
  const at = new Date();
  test('certifies a page only once every current chunk was embedded this pass', () => {
    const cov = new Map<string, { shapeId: string; covered: Set<number> }>();
    expect(pageFullyEmbeddedThisPass(cov, 'k', 'S', [0, 1], [
      { chunk_index: 0, embedded_at: null }, { chunk_index: 1, embedded_at: null }, { chunk_index: 2, embedded_at: null },
    ])).toBe(false);
    expect(pageFullyEmbeddedThisPass(cov, 'k', 'S', [2], [
      { chunk_index: 0, embedded_at: at }, { chunk_index: 1, embedded_at: at }, { chunk_index: 2, embedded_at: null },
    ])).toBe(true);
  });
  test('a page re-imported mid-pass is not certified', () => {
    const cov = new Map<string, { shapeId: string; covered: Set<number> }>();
    pageFullyEmbeddedThisPass(cov, 'k', 'S', [0, 1], [
      { chunk_index: 0, embedded_at: null }, { chunk_index: 1, embedded_at: null },
      { chunk_index: 2, embedded_at: null }, { chunk_index: 3, embedded_at: null },
    ]);
    // Re-import replaced four chunks with three NULL ones behind the cursor.
    expect(pageFullyEmbeddedThisPass(cov, 'k', 'S', [2], [
      { chunk_index: 0, embedded_at: null }, { chunk_index: 1, embedded_at: null }, { chunk_index: 2, embedded_at: null },
    ])).toBe(false);
  });
  test('chunks embedded before this pass do not count as covered', () => {
    const cov = new Map<string, { shapeId: string; covered: Set<number> }>();
    expect(pageFullyEmbeddedThisPass(cov, 'k', 'S', [1], [
      { chunk_index: 0, embedded_at: at }, { chunk_index: 1, embedded_at: null },
    ])).toBe(false);
  });

  test('a shape change mid-pass restarts coverage (retitled page is not certified)', () => {
    const cov = new Map<string, { shapeId: string; covered: Set<number> }>();
    pageFullyEmbeddedThisPass(cov, 'k', 'title-A', [0], [
      { chunk_index: 0, embedded_at: null }, { chunk_index: 1, embedded_at: null },
    ]);
    expect(pageFullyEmbeddedThisPass(cov, 'k', 'title-B', [1], [
      { chunk_index: 0, embedded_at: at }, { chunk_index: 1, embedded_at: null },
    ])).toBe(false);
  });
  test('certified pages are evicted and pruning keeps only the continuing page', () => {
    const cov = new Map<string, { shapeId: string; covered: Set<number> }>();
    expect(pageFullyEmbeddedThisPass(cov, 'a', 'S', [0], [{ chunk_index: 0, embedded_at: null }])).toBe(true);
    expect(cov.has('a')).toBe(false);
    pageFullyEmbeddedThisPass(cov, 'b', 'S', [0], [{ chunk_index: 0, embedded_at: null }, { chunk_index: 1, embedded_at: null }]);
    pageFullyEmbeddedThisPass(cov, 'c', 'S', [0], [{ chunk_index: 0, embedded_at: null }, { chunk_index: 1, embedded_at: null }]);
    prunePassCoverage(cov, 'c');
    expect([...cov.keys()]).toEqual(['c']);
  });
});
