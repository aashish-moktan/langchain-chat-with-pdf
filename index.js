import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PineconeStore } from "@langchain/pinecone";
import { pinecone, googleGenAIEmbeddings } from "./config";

async function loadPDF(pdfPath) {
  const pdfLoader = new PDFLoader(pdfPath);
  const rawDocs = await pdfLoader.load();
  return rawDocs;
}

async function indexDocument() {
  // step 1: load the pdf file
  const PDF_PATH = "./dsa.pdf";
  const rawDocs = await loadPDF(PDF_PATH);
  console.log("PDF loaded");

  // step 2: create the chunks of pdf
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
  console.log("Chunking completed");

  // step 3: initialize pinecone index
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
  console.log("Pinecone configured");

  // step 4: langchain (chunking, embeding, database)
  await PineconeStore.fromDocuments(chunkedDocs, googleGenAIEmbeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });
  console.log("Data stored successfully");
}

indexDocument();
