"use client";

import Link from "next/link";
import Brand from "../components/Brand";
import { ArrowPathIcon } from "@heroicons/react/24/outline";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return <main className="system-page"><Brand/><div><h1>Something didn’t load.</h1><p>Try opening the page again. Your saved sessions remain in this browser.</p><button className="system-primary" onClick={reset}><ArrowPathIcon/> Try again</button><Link className="system-secondary" href="/">Back to home</Link></div></main>;
}
