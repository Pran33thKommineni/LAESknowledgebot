import dotenv from "dotenv";

dotenv.config();

export const config = {
  port: Number(process.env.PORT) || 4000,
  groqApiKey: process.env.GROQ_API_KEY || "",
  // Optional: legacy OneDrive/Graph config (not needed if using local folder sync)
  onedriveRootFolderId: process.env.ONEDRIVE_ROOT_FOLDER_ID || "",
  msTenantId: process.env.MS_TENANT_ID || "",
  msClientId: process.env.MS_CLIENT_ID || "",
  msClientSecret: process.env.MS_CLIENT_SECRET || "",
  // Preferred: sync from a locally-available LAES folder
  localLaesFolder: process.env.LOCAL_LAES_FOLDER || "",
  dbPath: process.env.DB_PATH || "data/laes-knowledge.db",
};

if (!config.groqApiKey) {
  // eslint-disable-next-line no-console
  console.warn("GROQ_API_KEY not set; QA pipeline will not work until configured.");
}
if (!config.localLaesFolder && !config.onedriveRootFolderId) {
  // eslint-disable-next-line no-console
  console.warn("No LOCAL_LAES_FOLDER or ONEDRIVE_ROOT_FOLDER_ID set; sync will fail until configured.");
}

