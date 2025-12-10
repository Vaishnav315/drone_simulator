import os
import json
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP ---
# Ensure you have the DATABASE_URL env var set in Render
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Fix for Render's postgres:// URL (SQLAlchemy needs postgresql://)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class EnemyDetection(Base):
        __tablename__ = "enemy_detections"
        id = Column(Integer, primary_key=True, index=True)
        drone_id = Column(String)
        enemy_type = Column(String)
        confidence = Column(Float)
        lat = Column(Float)
        long = Column(Float)
        detected_at = Column(DateTime, default=datetime.datetime.utcnow)

    Base.metadata.create_all(bind=engine)
else:
    SessionLocal = None
    print("WARNING: No Database URL found.")

# --- LOAD YOUR CUSTOM BRAIN ---
model_filename = "drone_model.pt"
if os.path.exists(model_filename):
    print(f"✅ Loading Custom VisDrone Model: {model_filename}")
    model = YOLO(model_filename)
else:
    print("⚠️ Custom model not found, falling back to generic yolov8n.pt")
    model = YOLO('yolov8n.pt')

@app.post("/ingest_drone_feed")
async def ingest_feed(
    file: UploadFile, 
    telemetry: str = Form(...) 
):
    try:
        # 1. Parse Data
        data = json.loads(telemetry)
        drone_lat = data.get("lat", 0.0)
        drone_long = data.get("long", 0.0)
        drone_id = data.get("drone_id", "Unknown")

        # 2. Save Temp Image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            img_path = tmp.name

        # 3. RUN AI (Inference)
        # conf=0.4: Only report if 40% sure
        results = model(img_path, conf=0.4)
        
        detections = []
        db = SessionLocal() if SessionLocal else None
        
        for result in results:
            for box in result.boxes:
                # Get Class ID and Name
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # VisDrone Classes often include: 
                # 0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck, 6: tricycle...
                # We filter for relevant military threats
                threats = ['pedestrian', 'people', 'car', 'truck', 'van', 'bus']
                
                if label in threats:
                    # Simple Geo-Location Math (Assumes object is near drone center)
                    # In a real app, you'd use altitude & pixel offset to calculate exact pos
                    offset_lat = np.random.uniform(-0.0002, 0.0002)
                    offset_long = np.random.uniform(-0.0002, 0.0002)
                    
                    obj_lat = drone_lat + offset_lat
                    obj_long = drone_long + offset_long
                    
                    det = {
                        "drone_id": drone_id,
                        "type": label,
                        "conf": conf,
                        "lat": obj_lat,
                        "long": obj_long
                    }
                    detections.append(det)

                    # Save to DB
                    if db:
                        new_threat = EnemyDetection(
                            drone_id=drone_id,
                            enemy_type=label,
                            confidence=conf,
                            lat=obj_lat,
                            long=obj_long
                        )
                        db.add(new_threat)

        if db:
            db.commit()
            db.close()
            
        os.remove(img_path)
        
        return {
            "status": "success",
            "threats_detected": len(detections),
            "data": detections
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/enemies")
def get_enemies():
    """API for Flutter Map to poll"""
    if not SessionLocal: return []
    db = SessionLocal()
    # Return last 100 detections
    data = db.query(EnemyDetection).order_by(EnemyDetection.detected_at.desc()).limit(100).all()
    db.close()
    return data
