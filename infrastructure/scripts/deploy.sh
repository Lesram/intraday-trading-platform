#!/bin/bash
# 🚀 MICROSERVICES DEPLOYMENT SCRIPT
# Complete infrastructure deployment with validation

set -e  # Exit on any error

echo "🚀 Starting Microservices Deployment..."
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed"
        exit 1
    fi
    
    print_success "All prerequisites met"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build all services
    docker build -f infrastructure/docker/Dockerfile.services --target prediction-service -t trading/prediction-service:latest .
    docker build -f infrastructure/docker/Dockerfile.services --target risk-service -t trading/risk-service:latest .
    docker build -f infrastructure/docker/Dockerfile.services --target data-service -t trading/data-service:latest .
    docker build -f infrastructure/docker/Dockerfile.services --target sentiment-service -t trading/sentiment-service:latest .
    docker build -f infrastructure/docker/Dockerfile.services --target portfolio-service -t trading/portfolio-service:latest .
    docker build -f infrastructure/docker/Dockerfile.services --target execution-service -t trading/execution-service:latest .
    docker build -f infrastructure/docker/Dockerfile.services --target monitoring-service -t trading/monitoring-service:latest .
    
    print_success "All Docker images built successfully"
}

# Deploy with Docker Compose (Development)
deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p data/postgres data/redis data/kafka logs
    
    # Start services
    docker-compose -f infrastructure/docker/docker-compose.yml up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    for service in postgres redis kafka kong; do
        if docker-compose -f infrastructure/docker/docker-compose.yml ps $service | grep -q "Up"; then
            print_success "$service is running"
        else
            print_error "$service failed to start"
        fi
    done
    
    print_success "Docker Compose deployment completed"
}

# Deploy to Kubernetes (Production)
deploy_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl apply -f infrastructure/kubernetes/monitoring-service.yaml
    
    # Deploy storage resources
    print_status "Creating persistent volumes..."
    kubectl apply -f infrastructure/kubernetes/monitoring-service.yaml
    
    # Deploy services in order
    services=(
        "data-service.yaml"
        "sentiment-service.yaml" 
        "prediction-service.yaml"
        "risk-service.yaml"
        "portfolio-service.yaml"
        "execution-service.yaml"
        "monitoring-service.yaml"
    )
    
    for service in "${services[@]}"; do
        print_status "Deploying $service..."
        kubectl apply -f infrastructure/kubernetes/$service
        sleep 10
    done
    
    # Wait for deployments to be ready
    print_status "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment --all -n trading
    
    print_success "Kubernetes deployment completed"
}

# Validate deployment
validate_deployment() {
    print_status "Validating deployment..."
    
    if [ "$1" = "kubernetes" ]; then
        # Check Kubernetes deployment
        print_status "Checking Kubernetes pods..."
        kubectl get pods -n trading
        
        # Check service endpoints
        print_status "Checking service endpoints..."
        kubectl get svc -n trading
        
        # Test API endpoints
        for service in prediction risk data sentiment portfolio execution monitoring; do
            if kubectl get svc ${service}-service -n trading &> /dev/null; then
                print_success "${service}-service is deployed"
            else
                print_error "${service}-service deployment failed"
            fi
        done
        
    else
        # Check Docker Compose deployment
        print_status "Checking Docker containers..."
        docker-compose -f infrastructure/docker/docker-compose.yml ps
        
        # Test API endpoints
        services=("prediction-service:8080" "risk-service:8080" "data-service:8080" "sentiment-service:8080" "portfolio-service:8080" "execution-service:8080" "monitoring-service:8080")
        
        for service in "${services[@]}"; do
            if curl -f http://localhost:${service##*:}/health &> /dev/null; then
                print_success "$service is healthy"
            else
                print_warning "$service health check failed"
            fi
        done
    fi
    
    print_success "Deployment validation completed"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring stack..."
    
    if [ "$1" = "kubernetes" ]; then
        # Port forward for monitoring access
        print_status "Setting up port forwards for monitoring..."
        kubectl port-forward svc/prometheus 9090:9090 -n monitoring &
        kubectl port-forward svc/grafana 3000:3000 -n monitoring &
        kubectl port-forward svc/jaeger-query 16686:16686 -n monitoring &
        
        print_success "Monitoring accessible at:"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000 (admin/admin)"
        echo "  - Jaeger: http://localhost:16686"
    else
        print_success "Monitoring accessible at:"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000 (admin/admin)"
        echo "  - Jaeger: http://localhost:16686"
        echo "  - Kong Manager: http://localhost:8002"
    fi
}

# Main deployment logic
main() {
    echo "🎯 Microservices Deployment Options:"
    echo "1. Docker Compose (Development)"
    echo "2. Kubernetes (Production)"
    echo ""
    
    read -p "Select deployment method (1 or 2): " choice
    
    case $choice in
        1)
            check_prerequisites
            build_images
            deploy_docker_compose
            validate_deployment "docker"
            setup_monitoring "docker"
            ;;
        2)
            check_prerequisites
            build_images
            deploy_kubernetes
            validate_deployment "kubernetes"
            setup_monitoring "kubernetes"
            ;;
        *)
            print_error "Invalid choice. Please select 1 or 2."
            exit 1
            ;;
    esac
    
    echo ""
    print_success "🎉 Microservices deployment completed successfully!"
    echo ""
    echo "📊 Next Steps:"
    echo "1. Configure API keys in secrets"
    echo "2. Set up Grafana dashboards"
    echo "3. Configure alerting rules"
    echo "4. Run integration tests"
    echo "5. Start shadow deployment validation"
    echo ""
    echo "🔧 Management Commands:"
    echo "  - View logs: kubectl logs -f deployment/prediction-service -n trading"
    echo "  - Scale services: kubectl scale deployment prediction-service --replicas=5 -n trading"
    echo "  - Update config: kubectl edit configmap trading-config -n trading"
    echo ""
}

# Run main function
main "$@"
