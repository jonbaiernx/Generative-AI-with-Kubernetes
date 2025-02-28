cat > adminuser-csr.yaml <<EOF
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: appxadmin
spec:
  groups:
  - system:authenticated
  request: $(cat adminuser-csr.pem | base64 | tr -d "\n")
  signerName: kubernetes.io/kube-apiserver-client
  usages:
  - client auth
EOF