pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        VENV = "${WORKSPACE}/.venv"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                sh '''
                python -m venv ${VENV}
                . ${VENV}/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Pipeline Compilation') {
            steps {
                sh '''
                . ${VENV}/bin/activate
                python pipeline.py
                '''
            }
        }
    }

    post {
        success {
            echo 'Jenkins pipeline completed successfully.'
        }
        failure {
            echo 'Jenkins pipeline failed. Check the logs.'
        }
        always {
            cleanWs()
        }
    }
}

