import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import http from "http";
import { createServer } from "http";
import { Server } from "socket.io";
import connectDB from "./config/db.js";
import authRoutes from "./routes/authRoutes.js";
import testRoutes from "./routes/testRoutes.js";

dotenv.config();
connectDB();

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
  },
});

app.use(express.json());
app.use(cors());

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/tests", testRoutes);

// Store active test sessions
const activeTests = new Map();

io.on("connection", (socket) => {
  console.log("A user connected:", socket.id);

  // Join a test session
  socket.on("join-test", ({ testId, userId }) => {
    console.log(testId,userId)
    socket.join(testId);
    if (!activeTests.has(testId)) {
      activeTests.set(testId, new Map());
    }
    activeTests.get(testId).set(socket.id, userId);

    io.to(testId).emit("update-participants", Array.from(activeTests.get(testId).values()));

    console.log(`User ${userId} joined test ${testId}`);
  });

  // Start test timer
  socket.on("start-test", ({ testId, duration }) => {
    io.to(testId).emit("test-started", { duration });
    console.log(`Test ${testId} started with duration ${duration} minutes`);
  });

  // Submit answers
  socket.on("submit-answer", ({ testId, userId, answers }) => {
    io.to(testId).emit("answer-submitted", { userId, answers });
    console.log(`User ${userId} submitted answers for test ${testId}`);
  });

  // Handle disconnection
  socket.on("disconnect", () => {
    activeTests.forEach((users, testId) => {
      if (users.has(socket.id)) {
        users.delete(socket.id);
        io.to(testId).emit("update-participants", Array.from(users.values()));
      }
    });

    console.log("A user disconnected:", socket.id);
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
