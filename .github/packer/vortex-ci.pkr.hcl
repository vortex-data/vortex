packer {
  required_plugins {
    amazon = {
      version = ">= 1.3.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "eu-west-1"
}

variable "arch" {
  type        = string
  description = "Architecture: x64 or arm64"
}

variable "ami_prefix" {
  type    = string
  default = "vortex-ci"
}

variable "source_ami_owner" {
  type        = string
  default     = "135269210855"
  description = "runs-on AWS account ID"
}

variable "subnet_id" {
  type    = string
  default = ""
}

variable "security_group_id" {
  type        = string
  default     = ""
  description = "Existing security group ID (must allow SSH inbound)"
}

variable "rust_toolchain" {
  type    = string
  default = "1.89"
}

variable "protoc_version" {
  type    = string
  default = "29.3"
}

variable "flatc_version" {
  type    = string
  default = "25.9.23"
}

locals {
  timestamp = formatdate("YYYYMMDD-HHmmss", timestamp())

  arch_config = {
    x64 = {
      instance_type   = "m7i.large"
      source_ami_name = "runs-on-v2.2-ubuntu24-full-x64-*"
      ami_arch        = "x86_64"
    }
    arm64 = {
      instance_type   = "m7g.large"
      source_ami_name = "runs-on-v2.2-ubuntu24-full-arm64-*"
      ami_arch        = "arm64"
    }
  }

  config = local.arch_config[var.arch]
}

source "amazon-ebs" "vortex-ci" {
  ami_name      = "${var.ami_prefix}-${var.arch}-${local.timestamp}"
  instance_type = local.config.instance_type
  region        = var.aws_region

  source_ami_filter {
    filters = {
      name                = local.config.source_ami_name
      root-device-type    = "ebs"
      virtualization-type = "hvm"
      architecture        = local.config.ami_arch
    }
    most_recent = true
    owners      = [var.source_ami_owner]
  }

  subnet_id         = var.subnet_id != "" ? var.subnet_id : null
  security_group_id = var.security_group_id != "" ? var.security_group_id : null
  ssh_username      = "runner"

  # User data to start SSH for Packer connectivity
  user_data_file = "${path.root}/scripts/user_data.sh"

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 80
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name        = "${var.ami_prefix}-${var.arch}"
    Environment = "ci"
    Arch        = var.arch
    BuildTime   = local.timestamp
    ManagedBy   = "packer"
  }
}

build {
  sources = ["source.amazon-ebs.vortex-ci"]

  # Run the provisioning script
  provisioner "shell" {
    script = "${path.root}/scripts/provision.sh"
    environment_vars = [
      "RUST_TOOLCHAIN=${var.rust_toolchain}",
      "PROTOC_VERSION=${var.protoc_version}",
      "FLATC_VERSION=${var.flatc_version}"
    ]
  }
}
