// Deploy bootstrap. When the deploy payload already contains the full talk/
// source (normal `vercel deploy` from talk/), this is a no-op. When only the
// bootstrap payload was uploaded (MCP/API deploys), it pulls the talk/ tree
// from the public repo's talk-site branch before install.
import { execSync } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';

if (existsSync('app/page.tsx')) {
  console.log('full source present — skipping fetch');
  process.exit(0);
}
const REF = process.env.ACD_SRC_REF || 'refs/heads/talk-site';
const url = `https://codeload.github.com/SharathSPhD/ActiveCIrcuitDiscovery/tar.gz/${REF}`;
const res = await fetch(url);
if (!res.ok) throw new Error(`tarball fetch failed: ${res.status}`);
writeFileSync('/tmp/src.tgz', Buffer.from(await res.arrayBuffer()));
execSync(`tar -xzf /tmp/src.tgz --strip-components=2 --wildcards "*/talk"`, { stdio: 'inherit' });
console.log(`talk/ source extracted from ${REF}`);
