# Base có sẵn torch + torchvision + CUDA
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .                      # chỉ code (data/.venv/repo phụ đã bị .dockerignore loại)

# KHÔNG bake data (mount volume) và KHÔNG bake WANDB key (truyền -e lúc run)
CMD ["bash"]
