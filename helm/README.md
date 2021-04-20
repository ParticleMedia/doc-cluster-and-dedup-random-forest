## List existing release
```
helm ls -n serving
```

## Upgrade or install application
```
helm upgrade -i spider-online-goproxy helm/goproxy \
    -f helm/goproxy/values_base.yaml \
    -f helm/goproxy/values_spider_online.yaml \
    --set server.image.tag=$TAG \
    --namespace serving \
    --dry-run --debug
```

## Delete existing deployment
```
helm delete spider-online-goproxy -n serving
```
