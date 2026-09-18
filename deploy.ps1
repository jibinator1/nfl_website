Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "DEPLOYING NFL ANALYTICS HUB TO VERCEL" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""
npx vercel --prod
Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Deployment complete!" -ForegroundColor Green
