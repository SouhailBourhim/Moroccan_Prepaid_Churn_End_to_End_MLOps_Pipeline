/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CHURN_API_URL?: string;
  // Sent as X-API-Key. Vite inlines it into the JS bundle, so only use the
  // dashboard against local or internal deployments with this set.
  readonly VITE_CHURN_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
