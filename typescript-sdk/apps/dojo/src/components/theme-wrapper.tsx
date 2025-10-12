"use client";

import { ThemeProvider } from "@/components/theme-provider";
import { useURLParams } from "@/contexts/url-params-context";

export function ThemeWrapper({ children }: { children: React.ReactNode }) {
  const { theme } = useURLParams();

  return (
    <ThemeProvider
      attribute="class"
      forcedTheme={theme}
      enableSystem={false}
      themes={["light", "dark"]}
      disableTransitionOnChange
    >
      {children}
    </ThemeProvider>
  );
}

