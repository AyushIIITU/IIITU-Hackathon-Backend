import mongoose from "mongoose";

const TestSchema = new mongoose.Schema(
  {
    creator: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    title: { type: String, required: true },
    pdfUrl: { type: String, default: null },
    youtubeLink: { type: String, default: null },
    questions: [{ type: mongoose.Schema.Types.ObjectId, ref: "Question" }],
    isPublic: { type: Boolean, default: true },
    accessCode: { type: String, unique: true, sparse: true }, // Unique code for private tests
    participants: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }], // Track users who joined
  },
  { timestamps: true }
);

const Test = mongoose.model("Test", TestSchema);
export default Test;
