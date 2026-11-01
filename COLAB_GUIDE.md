# Using FaceLock_Colab in Google Colab

1. Upload `FaceLock_Colab.ipynb` to Google Drive and open it with Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Zip the complete `FaceLock_Colab` folder on your computer.
4. Run the notebook top to bottom. When asked, upload the ZIP and then an input image. The copied project includes sample images if you prefer to use one.
5. For protection, provide a Hugging Face read token with access to the required model repositories.

The notebook covers editing, FaceLock protection, editing the protected result, and pointers for batch jobs and evaluation. Colab storage is temporary unless you download outputs or copy them to Drive. See `README.md` for full project documentation.
