// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { WorkloadFigure } from '@/components/WorkloadFigure';
import { collectShowcaseClaims, type ShowcaseClaim } from '@/lib/synthesis';

// Rendered per request: every headline number is a live geomean computed by the
// same synthesis the Latest page renders, so a claim and its proof can never
// disagree. `force-dynamic` keeps `next build` DB-free.
export const dynamic = 'force-dynamic';

/**
 * The Overview tab (`/`): the "claim → why it matters → proof" showcase, the
 * React port of `server/src/html/showcase.rs::showcase_body`. Four
 * Vortex-vs-Parquet claims in a grid, each pairing a live headline geomean with
 * a blueprint schematic of the workload and a deep link to its proof on the
 * Latest page.
 */
export default async function Overview() {
  const claims = await collectShowcaseClaims();
  return (
    <section className="showcase">
      <header className="showcase-intro">
        <p className="showcase-eyebrow">Vortex vs Apache Parquet</p>
        <h2 className="showcase-headline">
          A columnar format built for the read patterns Parquet wasn&rsquo;t.
        </h2>
      </header>
      <div className="claims">
        {claims.map((claim) => (
          <ClaimCard key={claim.label} claim={claim} />
        ))}
      </div>
      <div className="showcase-cta">
        <a className="show-everything" href="/latest">
          Show me everything
          <span className="show-everything-arrow" aria-hidden="true">
            {' '}
            &rarr;
          </span>
        </a>
      </div>
    </section>
  );
}

/** One claim cell: the headline stat beside a workload schematic, with the "why
 * it matters" prose spanning the cell beneath. */
function ClaimCard({ claim }: { claim: ShowcaseClaim }) {
  return (
    <article className="claim">
      <div className="claim-head">
        <div className="claim-stat">
          <span className="claim-metric">{claim.hero}</span>
          <span className="claim-label">{claim.label}</span>
          {claim.detail !== null && <span className="claim-detail">{claim.detail}</span>}
          {claim.href !== null && (
            <a className="claim-proof" href={claim.href}>
              See the proof
              <span className="claim-proof-arrow" aria-hidden="true">
                {' '}
                &rarr;
              </span>
            </a>
          )}
        </div>
        <WorkloadFigure workload={claim.workload} />
      </div>
      <p className="claim-why">{claim.why}</p>
    </article>
  );
}
