# apply deployement
kubectl apply -f https://k8s.io/examples/service/load-balancer-example.yaml
kubectl apply -f surfnet.yaml

# get deployment information
kubectl get deployments <deployment-name>
kubectl describe deployments <deployment-name>

# get replicaset information
kubectl get replicasets
kubectl describe replicasets

# get pod information
kubectl get pods --output=wide
kubectl get pod <pod-name>
kubectl describe pod <pod-name>

# Log in k8s running container
POD_NAME=<pod-name>
CONTAINER_NAME=surfnet
kubectl exec -it ${POD_NAME} -c ${CONTAINER_NAME} -- /bin/bash

# pod Logs
kubectl logs <pod-name>
kubectl attach <pod-name>

# cluster info
kubectl cluster-info dump

# create service
kubectl expose deployment hello-world --type=LoadBalancer --name=hello-api
kubectl expose deployment surfnet --type=LoadBalancer --name=surfnet-api

# get service information
kubectl get services
kubectl describe services <service-name>

# delete service
kubectl delete services <service-name>

# node-pools info
kubectl get nodes --show-labels
