# Fashion Recommender System
An AI-powered image-based recommendation engine that suggests visually similar fashion items using deep learning and vector similarity search.

### Project Summary
Built a Streamlit web app that recommends similar fashion items based on an uploaded image.

Extracts deep features using EfficientNetB3, and performs fast similarity search using FAISS over 50,000+ fashion images hosted on AWS S3.

Achieves 88% Precision@5 with average latency of 0.81s, outperforming a baseline ResNet50 + KNN model (Precision@5 of 74%).

### Try the App
Launch Web App (replace with actual deployment link)

How It Works
 - User uploads a fashion image (shirt, dress, bag, accessories, etc.)

 - Image is preprocessed and passed through a pretrained CNN (EfficientNetB3 or ResNet50)

 - Extracted embeddings are normalized and searched against a vector index

 - Top 5 similar fashion images are returned using L2 similarity

### Performance Comparison
| Model          | Retrieval Method  | Precision\@5 | Avg Latency |
| -------------- | ----------------- | ------------ | ----------- |
| EfficientNetB3 | FAISS (L2 Index)  | **88%**      | **0.81s**   |
| ResNet50       | KNN (Brute Force) | 74%          |   0.74s     |


EfficientNetB3+FAISS version is more accurate than the baseline.

### Features
 - Upload any fashion item image (RGB/JPEG/PNG)

 - Top 5 similar items visualized with download or share-ready thumbnails

 - Efficient vector search using FAISS

 - Backend hosted embeddings from 50K+ fashion items

### Potential Applications
 - E-commerce: “Visually similar” suggestions on product pages

 - Style inspiration: Users find alternatives based on design/patterns

 - Fashion curation: Designers analyze trends across collections

 - Marketplace UX: Personalized suggestions for better conversion

### What I Built
Extracted deep visual features using EfficientNetB3 and ResNet50

Compared KNN+ ResNet vs. FAISS retrieval with EfficientNetB3 for speed and accuracy

Precomputed and saved embeddings for 50K fashion images

Built an interactive Streamlit app for demo and testing using EfficientNetB3+FAISS version due to its accuracy

### Tech Stack
Python, Streamlit, NumPy, FAISS, Keras, TensorFlow

EfficientNetB3, ResNet50

AWS S3 (public buckets for image hosting)

Pickle for embedding and metadata storage

### Project Structure

fashion_recommender/
├── app_using_EfficientNet_faiss.py                          # Streamlit app with EfficientNetB3 + FAISS
├── baseline_resnet_knn.py                                   # ResNet50 + KNN baseline script
├── all_embeddings_efficient_max.pkl                         # Embeddings generated on 50k images using EfficientNet
├── all_embeddings_resnet_max.pkl                            # Embeddings generated on 50k images using ResNet
├── filenames.pkl                                            # AWS S3 URLs for fashion items ( images )
├── requirements.txt
├── README.md
└── .gitignore















 
      

