import cv2
import mediapipe as mp
import os
from tkinter import *
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create dataset directory
os.makedirs("dataset", exist_ok=True)

class ASLCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sign Capture")
        self.sign_label = simpledialog.askstring("Sign Label", "Enter the ASL sign to capture:")
        if not self.sign_label:
            messagebox.showerror("Error", "Sign label is required!")
            root.destroy()
            return

        self.capture_count = 0
        self.max_images = 100
        self.dataset_dir = os.path.join("dataset", self.sign_label)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.video_label = Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and self.capture_count < self.max_images:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save frame
                img_name = f"{self.capture_count:04d}.jpg"
                img_path = os.path.join(self.dataset_dir, img_name)
                cv2.imwrite(img_path, frame)
                self.capture_count += 1

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if self.capture_count >= self.max_images:
            self.cap.release()
            messagebox.showinfo("Done", f"Captured {self.max_images} images for sign '{self.sign_label}'")
            self.root.destroy()
            return

        self.root.after(10, self.update_video)

if __name__ == "__main__":
    root = Tk()
    app = ASLCaptureApp(root)
    root.mainloop()
