import express from "express";
import { createTest, joinTest } from "../controllers/testController.js";
import authMiddleware from "../middlewares/auth.js";
import upload from "../middlewares/upload.js";

const router = express.Router();

// Route to create a test with PDF or YouTube link
router.post("/create", authMiddleware, upload.single("pdf"), createTest);
router.post("/join", authMiddleware, joinTest);
export default router;
