import Test from "../models/Test.js";
import crypto from "crypto";
import User from "../models/User.js";


export const createTest = async (req, res) => {
  try {
    const { title, isPublic, youtubeLink } = req.body;
    const pdfUrl = req.file ? `/uploads/${req.file.filename}` : null;

    // Generate a unique 6-character access code for private tests
    const accessCode = isPublic ? null : crypto.randomBytes(3).toString("hex").toUpperCase();

    const test = new Test({
      creator: req.user.id,
      title,
      pdfUrl,
      youtubeLink,
      isPublic,
      accessCode, // Set the generated accessCode
    });

    await test.save();
    
    res.status(201).json({
      message: "Test created successfully",
      test,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Server Error", error });
  }
};

export const joinTest = async (req, res) => {
    try {
      const { testId, accessCode } = req.body;
      const test = await Test.findById(testId);
  
      if (!test) return res.status(404).json({ message: "Test not found" });
  
      if (!test.isPublic && test.accessCode !== accessCode)
        return res.status(403).json({ message: "Invalid access code" });
  
      if (!test.participants.includes(req.user.id)) {
        test.participants.push(req.user.id);
        await test.save();
      }
  
      res.json({ message: "Joined test successfully", test });
    } catch (error) {
      res.status(500).json({ message: "Server Error", error });
    }
  };