import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { CaseView } from "@/components/case/case-view";
import { DEFAULT_PERSONA } from "@/lib/token";

export const dynamic = "force-dynamic";

export default async function CasePage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ persona?: string }>;
}) {
  const { id } = await params;
  const persona = (await searchParams).persona ?? DEFAULT_PERSONA;

  let detail;
  try {
    detail = await api.case(id, persona);
  } catch (error) {
    return <NotFound id={id} error={error} />;
  }

  return <CaseView detail={detail} persona={persona} />;
}

function NotFound({ id, error }: { id: string; error: unknown }) {
  const status = error instanceof ApiError ? error.status : 0;
  const message = error instanceof ApiError ? error.message : String(error);

  return (
    <div className="rounded-lg border border-paper-edge bg-paper p-6">
      <h1 className="text-lg font-semibold text-ink">
        {status === 403 ? "You cannot read this case." : `No case ${id}.`}
      </h1>
      <p className="mt-2 text-sm text-ink-muted">{message}</p>
      {status === 403 && (
        <p className="mt-2 text-sm text-ink-muted">
          A case is readable by a persona the KPI&rsquo;s access policy would
          have let run it. That is the row filter, not a UI decision — try a
          different reader.
        </p>
      )}
      <Link
        href="/"
        className="mt-4 inline-block text-sm text-accent hover:underline"
      >
        ← Back to the watchlist
      </Link>
    </div>
  );
}
