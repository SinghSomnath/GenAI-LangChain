# Configure the AWS Provider
provider "aws" {
  region = "us-west-1"  # Change to your preferred region
  profile = "aws-dev" #Terraform to use the credentials and settings associated with the IAM user "aws-dev" >>> whose  AWS credentials file (usually located at ~/.aws/credentials).
}

# Create an S3 bucket
resource "aws_s3_bucket" "example_bucket" {
  bucket = "my-terraform-example-bucket-041083"  # Must be globally unique, so add your initials or a number if needed
  tags = {
    Name        = "My Example Bucket"
    Environment = "Dev"
  }
}