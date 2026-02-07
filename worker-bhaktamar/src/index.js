/**
 * Bhaktamar Voice Bot — Cloudflare Worker
 *
 * Proxies /api/* to the Bhaktamar Python backend on EC2 (port 8081).
 * Frontend is served from GitHub Pages.
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }

    // Proxy /api/* to Python backend
    if (url.pathname.startsWith('/api/')) {
      const backendUrl = env.BACKEND_URL || 'http://localhost:8081';
      const backendPath = url.pathname.replace('/api/', '/');
      const target = backendUrl + backendPath + url.search;

      try {
        const resp = await fetch(target, {
          method: request.method,
          headers: {
            'Content-Type': 'application/json',
          },
          body: request.method !== 'GET' ? request.body : undefined,
        });

        const body = await resp.text();
        return new Response(body, {
          status: resp.status,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
          },
        });
      } catch (err) {
        return new Response(
          JSON.stringify({ error: 'Backend unreachable', detail: err.message }),
          {
            status: 502,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*',
            },
          }
        );
      }
    }

    // Redirect root to GitHub Pages
    return new Response('Bhaktamar Voice Bot API — use /api/* endpoints', {
      headers: {
        'Content-Type': 'text/plain',
        'Access-Control-Allow-Origin': '*',
      },
    });
  },
};
