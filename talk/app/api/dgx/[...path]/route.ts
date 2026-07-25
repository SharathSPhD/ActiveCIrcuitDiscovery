import { NextRequest } from 'next/server';

/* Proxy to the DGX Spark FastAPI backend over the Cloudflare tunnel.
   The tunnel URL and API key live in server-side env vars, never in the client.
   Streams SSE bodies straight through. */

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

const ALLOWED = new Set(['health', 'prompts', 'graph', 'episode', 'steer']);

async function proxy(req: NextRequest, params: { path: string[] }) {
  const base = process.env.DGX_TUNNEL_URL;
  if (!base) {
    return Response.json({ error: 'offline', detail: 'DGX_TUNNEL_URL not configured' }, { status: 503 });
  }
  const path = params.path.join('/');
  if (!ALLOWED.has(params.path[0])) {
    return Response.json({ error: 'not found' }, { status: 404 });
  }
  const url = `${base.replace(/\/$/, '')}/${path}`;
  const headers: Record<string, string> = { 'x-acd-key': process.env.DGX_API_KEY ?? '' };
  const init: RequestInit = { method: req.method, headers, signal: AbortSignal.timeout(280_000) };
  if (req.method === 'POST') {
    headers['content-type'] = 'application/json';
    init.body = await req.text();
  }
  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (e: any) {
    return Response.json({ error: 'offline', detail: String(e?.message ?? e) }, { status: 503 });
  }
  const respHeaders = new Headers();
  const ct = upstream.headers.get('content-type') ?? 'application/json';
  respHeaders.set('content-type', ct);
  if (ct.includes('text/event-stream')) {
    respHeaders.set('cache-control', 'no-cache');
    respHeaders.set('x-accel-buffering', 'no');
  }
  return new Response(upstream.body, { status: upstream.status, headers: respHeaders });
}

export async function GET(req: NextRequest, ctx: { params: { path: string[] } }) {
  return proxy(req, ctx.params);
}
export async function POST(req: NextRequest, ctx: { params: { path: string[] } }) {
  return proxy(req, ctx.params);
}
