/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include "avm_gpu/pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct avm_gpu_pipeline {
  VkShaderModule module;
  VkDescriptorSetLayout dsl;
  VkPipelineLayout layout;
  VkPipeline pipeline;
  VkDescriptorPool pool;
  VkDescriptorSet ds;
  VkCommandBuffer cb;
  int n_bindings;
};

static int read_file(const char *path, void **data, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long n = ftell(f);
  if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -1;
  }
  void *buf = malloc((size_t)n);
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
    free(buf);
    fclose(f);
    return -1;
  }
  fclose(f);
  *data = buf;
  *size = (size_t)n;
  return 0;
}

int avm_gpu_pipeline_create(avm_gpu_ctx *ctx, const char *spv_path_default,
                            const char *spv_env, int n_bindings,
                            size_t push_constants_size,
                            avm_gpu_pipeline **out,
                            char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  if (n_bindings <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "n_bindings must be positive");
    return -1;
  }
  avm_gpu_pipeline *p = calloc(1, sizeof(*p));
  if (!p) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  p->n_bindings = n_bindings;
  const VkDevice dev = avm_gpu_device(ctx);

  const char *path = spv_path_default;
  if (spv_env) {
    const char *env = getenv(spv_env);
    if (env && *env) path = env;
  }

  void *code = NULL;
  size_t code_size = 0;
  if (read_file(path, &code, &code_size) != 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "cannot read SPIR-V '%s'", path);
    free(p);
    return -1;
  }
  VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  smci.codeSize = code_size;
  smci.pCode = (const uint32_t *)code;
  VkResult r = vkCreateShaderModule(dev, &smci, NULL, &p->module);
  free(code);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateShaderModule (%d)", (int)r);
    free(p);
    return -1;
  }

  VkDescriptorSetLayoutBinding *bind = calloc((size_t)n_bindings, sizeof(*bind));
  if (!bind) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    avm_gpu_pipeline_destroy(ctx, p);
    return -1;
  }
  for (int i = 0; i < n_bindings; i++) {
    bind[i].binding = (uint32_t)i;
    bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind[i].descriptorCount = 1;
    bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
  };
  dslci.bindingCount = (uint32_t)n_bindings;
  dslci.pBindings = bind;
  r = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &p->dsl);
  free(bind);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateDescriptorSetLayout (%d)", (int)r);
    avm_gpu_pipeline_destroy(ctx, p);
    return -1;
  }

  VkPushConstantRange pcr;
  memset(&pcr, 0, sizeof(pcr));
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.size = (uint32_t)push_constants_size;
  VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p->dsl;
  plci.pushConstantRangeCount = push_constants_size ? 1 : 0;
  plci.pPushConstantRanges = &pcr;
  r = vkCreatePipelineLayout(dev, &plci, NULL, &p->layout);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreatePipelineLayout (%d)", (int)r);
    avm_gpu_pipeline_destroy(ctx, p);
    return -1;
  }

  VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = p->module;
  cpci.stage.pName = "main";
  cpci.layout = p->layout;
  r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &p->pipeline);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateComputePipelines (%d)", (int)r);
    avm_gpu_pipeline_destroy(ctx, p);
    return -1;
  }

  VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              (uint32_t)n_bindings };
  VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  r = vkCreateDescriptorPool(dev, &dpci, NULL, &p->pool);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateDescriptorPool (%d)", (int)r);
    avm_gpu_pipeline_destroy(ctx, p);
    return -1;
  }

  VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
  dsai.descriptorPool = p->pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p->dsl;
  r = vkAllocateDescriptorSets(dev, &dsai, &p->ds);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkAllocateDescriptorSets (%d)", (int)r);
    avm_gpu_pipeline_destroy(ctx, p);
    return -1;
  }

  VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cbai.commandPool = avm_gpu_command_pool(ctx);
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  r = vkAllocateCommandBuffers(dev, &cbai, &p->cb);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkAllocateCommandBuffers (%d)", (int)r);
    avm_gpu_pipeline_destroy(ctx, p);
    return -1;
  }

  *out = p;
  return 0;
}

void avm_gpu_pipeline_destroy(avm_gpu_ctx *ctx, avm_gpu_pipeline *p) {
  if (!p) return;
  const VkDevice dev = avm_gpu_device(ctx);
  if (p->cb) vkFreeCommandBuffers(dev, avm_gpu_command_pool(ctx), 1, &p->cb);
  if (p->pool) vkDestroyDescriptorPool(dev, p->pool, NULL);
  if (p->pipeline) vkDestroyPipeline(dev, p->pipeline, NULL);
  if (p->layout) vkDestroyPipelineLayout(dev, p->layout, NULL);
  if (p->dsl) vkDestroyDescriptorSetLayout(dev, p->dsl, NULL);
  if (p->module) vkDestroyShaderModule(dev, p->module, NULL);
  free(p);
}

int avm_gpu_pipeline_dispatch(avm_gpu_ctx *ctx, avm_gpu_pipeline *p,
                              const avm_gpu_buf *bufs, int n_bindings,
                              const void *pc, size_t pc_size, uint32_t groups,
                              char err[AVM_GPU_ERR_LEN]) {
  if (n_bindings != p->n_bindings || !bufs) {
    snprintf(err, AVM_GPU_ERR_LEN, "binding count mismatch");
    return -1;
  }
  const VkDevice dev = avm_gpu_device(ctx);

  VkDescriptorBufferInfo *info = calloc((size_t)n_bindings, sizeof(*info));
  VkWriteDescriptorSet *wr = calloc((size_t)n_bindings, sizeof(*wr));
  if (!info || !wr) {
    free(info);
    free(wr);
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  for (int i = 0; i < n_bindings; i++) {
    info[i].buffer = bufs[i].buffer;
    info[i].offset = 0;
    info[i].range = VK_WHOLE_SIZE;
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = p->ds;
    wr[i].dstBinding = (uint32_t)i;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &info[i];
  }
  vkUpdateDescriptorSets(dev, (uint32_t)n_bindings, wr, 0, NULL);
  free(info);
  free(wr);

  VkCommandBuffer cb = p->cb;
  vkResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bi);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1,
                          &p->ds, 0, NULL);
  if (pc && pc_size) {
    vkCmdPushConstants(cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       (uint32_t)pc_size, pc);
  }
  vkCmdDispatch(cb, groups, 1, 1);
  vkEndCommandBuffer(cb);

  return avm_gpu_submit_wait(ctx, cb, err);
}
