import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
});

export const getChats = () => API.get("/chats");

export const createChat = (chat_name) =>
  API.post("/chats", { chat_name: chat_name || null });

export const getFiles = (chatId) =>
  API.get(`/chats/${chatId}/files`);

export const getMessages = (chatId) =>
  API.get(`/chats/${chatId}/messages`);

export const saveMessage = (chatId, data) =>
  API.post(`/chats/${chatId}/messages`, data);

export const uploadFiles = (chatId, files) => {
  const formData = new FormData();

  files.forEach((file) => {
    formData.append("files", file);
  });

  return API.post(`/chat/${chatId}/upload`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    timeout: 120000,
  });
};

export const queryChat = (chatId, query, intelligence) =>
  API.post(
    `/chat/${chatId}/query`,
    { query, intelligence },
    { timeout: 60000 }
  );

/**
 * Request the Chat Summary Report PDF from the backend and trigger a browser
 * file download. Uses responseType:'blob' so axios treats the binary PDF bytes
 * correctly instead of trying to parse them as JSON.
 */
export const downloadSummaryReport = async (chatId) => {
  const response = await API.post(
    `/chat/${chatId}/summary`,
    {},
    {
      responseType: "blob",
      timeout: 120000,
    }
  );

  // Extract filename from Content-Disposition header when available,
  // otherwise fall back to a sensible default.
  const disposition = response.headers["content-disposition"] || "";
  const filenameMatch = disposition.match(/filename="?([^";\n]+)"?/);
  const filename = filenameMatch
    ? filenameMatch[1]
    : `chat-summary-${chatId}.pdf`;

  // Programmatically click a temporary anchor to trigger browser download
  const url = window.URL.createObjectURL(
    new Blob([response.data], { type: "application/pdf" })
  );
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
};