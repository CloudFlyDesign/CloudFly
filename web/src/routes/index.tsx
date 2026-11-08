import { createFileRoute } from "@tanstack/react-router";
import TownApp from "@/game/TownApp";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  return <TownApp />;
}
