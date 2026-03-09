import { Client } from "@microsoft/microsoft-graph-client";
import { ClientSecretCredential } from "@azure/identity";
import "isomorphic-fetch";
import { config } from "./config";

// Simple Graph client using client credentials flow.
function createGraphClient(): Client {
  if (!config.msTenantId || !config.msClientId || !config.msClientSecret) {
    throw new Error("Microsoft Graph credentials are not fully configured.");
  }

  const credential = new ClientSecretCredential(
    config.msTenantId,
    config.msClientId,
    config.msClientSecret,
  );

  const authProvider = {
    getAccessToken: async () => {
      const token = await credential.getToken("https://graph.microsoft.com/.default");
      return token?.token ?? "";
    },
  };

  return Client.initWithMiddleware({
    authProvider,
  });
}

export interface DriveItem {
  id: string;
  name: string;
  webUrl: string;
  size: number;
  lastModifiedDateTime: string;
  file?: unknown;
  folder?: unknown;
  parentReference?: {
    path?: string;
  };
}

export class OneDriveClient {
  private client: Client;

  constructor() {
    this.client = createGraphClient();
  }

  async listFolderItems(folderId: string): Promise<DriveItem[]> {
    const response = await this.client
      .api(`/me/drive/items/${folderId}/children`)
      .select("id,name,webUrl,size,lastModifiedDateTime,folder,file,parentReference")
      .get();
    return response.value as DriveItem[];
  }

  async downloadFileContent(itemId: string): Promise<Buffer> {
    const response = await this.client.api(`/me/drive/items/${itemId}/content`).get();
    if (Buffer.isBuffer(response)) {
      return response;
    }
    // Graph client may return ArrayBuffer-like
    return Buffer.from(response as ArrayBuffer);
  }
}

