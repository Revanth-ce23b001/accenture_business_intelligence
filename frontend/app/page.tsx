import { Suspense } from "react";
import { api, ApiError } from "@/lib/api";
import { KpiCard } from "@/components/watchlist/kpi-card";
import {
  OpenCasesPanel,
  SuppressedPanel,
} from "@/components/watchlist/suppressed-panel";
import { PersonaSwitcher } from "@/components/persona-switcher";
import { DEFAULT_PERSONA } from "@/lib/token";

export const dynamic = "force-dynamic";

/**
 * Screen 1 — the Watchlist.
 *
 * Six KPI cards, and then the panel that matters: what was checked and
 * NOT opened as a case, with the gate that stopped each. The order is
 * deliberate — the cards say what is being watched, the panel says what
 * the watching concluded.
 */
export default async function WatchlistPage({
  searchParams,
}: {
  searchParams: Promise<{ persona?: string; scan?: string }>;
}) {
  const params = await searchParams;
  const persona = params.persona ?? DEFAULT_PERSONA;
  const scan = params.scan === "true";

  let watchlist;
  try {
    watchlist = await api.watchlist(persona, scan);
  } catch (error) {
    return <Unreachable error={error} />;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">
            Watchlist
          </h1>
          <p className="mt-1 text-sm text-ink-muted">
            Every material movement, and what was decided about it.{" "}
            <span className="font-mono text-xs text-ink-faint">
              {watchlist.period}
            </span>
          </p>
        </div>
        <Suspense>
          <PersonaSwitcher persona={persona} />
        </Suspense>
      </div>

      <section
        aria-label="KPIs"
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6"
      >
        {watchlist.cards.map((card) => (
          <KpiCard key={card.kpi} card={card} />
        ))}
      </section>

      <SuppressedPanel
        items={watchlist.suppressed}
        period={watchlist.period}
        scanned={watchlist.scanned}
      />

      <OpenCasesPanel items={watchlist.open_cases} />

      <p className="text-center font-mono text-[10px] text-ink-faint">
        Every figure above is computed by the engine and clickable to its
        evidence inside a case. Nothing on this page came from a model.
      </p>
    </div>
  );
}

function Unreachable({ error }: { error: unknown }) {
  const message = error instanceof ApiError ? error.message : String(error);
  const hint = error instanceof ApiError ? error.hint : undefined;
  return (
    <div className="rounded-lg border border-red-300 bg-red-50 p-6">
      <h1 className="text-lg font-semibold text-red-900">
        The watchlist could not be loaded.
      </h1>
      <p className="mt-2 text-sm text-red-800">{message}</p>
      {hint && <p className="mt-1 text-sm text-red-700">{hint}</p>}
      <p className="mt-4 text-xs text-red-700">
        Deliberately an error rather than an empty watchlist. A page that
        rendered &ldquo;nothing is wrong&rdquo; because the API was unreachable
        would be the worst failure this product could have.
      </p>
    </div>
  );
}
