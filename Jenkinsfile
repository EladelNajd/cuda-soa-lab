pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/EladelNajd/cuda-soa-lab.git'
            }
        }

        stage('Test CUDA') {
            steps {
                sh 'python3 test_cuda_kernel.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t gpu-service .'
            }
        }

        stage('Deploy') {
            steps {
                sh 'docker run --gpus all -d -p 5000:5000 gpu-service'
            }
        }
    }
}
