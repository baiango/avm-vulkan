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

#include "avm_gpu/vulkan_sad.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef AVM_GPU_SAD_SPV_PATH
#define AVM_GPU_SAD_SPV_PATH "sad.spv"
#endif

struct avm_gpu_sad {
  VkShaderModule module;
  VkDescriptorSetLayout dsl;
  VkPipelineLayout layout;
  VkPipeline pipeline;
  VkDescriptorPool pool;
  VkCommandBuffer cb;
  VkDescriptorSet ds;
  avm_gpu_buf src, ref, pos, cost;
  VkDeviceSize src_cap, ref_cap, pos_cap, cost_cap;
  int src_stride, ref_stride;
};

static int ensure_buf(avm_gpu_ctx *ctx, avm_gpu_buf *buf, VkDeviceSize *cap,
                      VkDeviceSize need, char err[AVM_GPU_ERR_LEN]) {
  if (*cap >= need) return 0;
  avm_gpu_buf_destroy(ctx, buf);
  if (avm_gpu_buf_create(ctx, need, buf, err) != 0) return -1;
  *cap = buf->size;
  return 0;
}

typedef struct push_constants {
  uint32_t bw, bh, src_stride, ref_stride, count;
} push_constants;

/* AVM_GPU_SAD_SPV overrides the compiled-in path; the build sets the default. */
static const char *spv_path(void) {
  const char *env = getenv("AVM_GPU_SAD_SPV");
  return env && *env ? env : AVM_GPU_SAD_SPV_PATH;
}

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

int avm_gpu_sad_create(avm_gpu_ctx *ctx, avm_gpu_sad **out,
                       char err[AVM_GPU_ERR_LEN]) {
  *out = NULL;
  avm_gpu_sad *sad = calloc(1, sizeof(*sad));
  if (!sad) {
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  const VkDevice dev = avm_gpu_device(ctx);

  void *code = NULL;
  size_t code_size = 0;
  if (read_file(spv_path(), &code, &code_size) != 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "cannot read SPIR-V '%s'", spv_path());
    free(sad);
    return -1;
  }
  VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  smci.codeSize = code_size;
  smci.pCode = (const uint32_t *)code;
  VkResult r = vkCreateShaderModule(dev, &smci, NULL, &sad->module);
  free(code);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateShaderModule (%d)", (int)r);
    free(sad);
    return -1;
  }

  VkDescriptorSetLayoutBinding bind[4];
  for (int i = 0; i < 4; i++) {
    memset(&bind[i], 0, sizeof(bind[i]));
    bind[i].binding = (uint32_t)i;
    bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind[i].descriptorCount = 1;
    bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
  };
  dslci.bindingCount = 4;
  dslci.pBindings = bind;
  r = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &sad->dsl);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateDescriptorSetLayout (%d)", (int)r);
    vkDestroyShaderModule(dev, sad->module, NULL);
    free(sad);
    return -1;
  }

  VkPushConstantRange pcr;
  memset(&pcr, 0, sizeof(pcr));
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.size = sizeof(push_constants);
  VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &sad->dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  r = vkCreatePipelineLayout(dev, &plci, NULL, &sad->layout);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreatePipelineLayout (%d)", (int)r);
    vkDestroyDescriptorSetLayout(dev, sad->dsl, NULL);
    vkDestroyShaderModule(dev, sad->module, NULL);
    free(sad);
    return -1;
  }

  VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = sad->module;
  cpci.stage.pName = "main";
  cpci.layout = sad->layout;
  r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &sad->pipeline);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateComputePipelines (%d)", (int)r);
    vkDestroyPipelineLayout(dev, sad->layout, NULL);
    vkDestroyDescriptorSetLayout(dev, sad->dsl, NULL);
    vkDestroyShaderModule(dev, sad->module, NULL);
    free(sad);
    return -1;
  }

  VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
  VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  r = vkCreateDescriptorPool(dev, &dpci, NULL, &sad->pool);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkCreateDescriptorPool (%d)", (int)r);
    vkDestroyPipeline(dev, sad->pipeline, NULL);
    vkDestroyPipelineLayout(dev, sad->layout, NULL);
    vkDestroyDescriptorSetLayout(dev, sad->dsl, NULL);
    vkDestroyShaderModule(dev, sad->module, NULL);
    free(sad);
    return -1;
  }

  VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
  dsai.descriptorPool = sad->pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &sad->dsl;
  r = vkAllocateDescriptorSets(dev, &dsai, &sad->ds);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkAllocateDescriptorSets (%d)", (int)r);
    avm_gpu_sad_destroy(ctx, sad);
    return -1;
  }

  VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cbai.commandPool = avm_gpu_command_pool(ctx);
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  r = vkAllocateCommandBuffers(dev, &cbai, &sad->cb);
  if (r != VK_SUCCESS) {
    snprintf(err, AVM_GPU_ERR_LEN, "vkAllocateCommandBuffers (%d)", (int)r);
    avm_gpu_sad_destroy(ctx, sad);
    return -1;
  }

  *out = sad;
  return 0;
}

void avm_gpu_sad_destroy(avm_gpu_ctx *ctx, avm_gpu_sad *sad) {
  if (!sad) return;
  const VkDevice dev = avm_gpu_device(ctx);
  if (sad->cb) {
    vkFreeCommandBuffers(dev, avm_gpu_command_pool(ctx), 1, &sad->cb);
  }
  avm_gpu_buf_destroy(ctx, &sad->src);
  avm_gpu_buf_destroy(ctx, &sad->ref);
  avm_gpu_buf_destroy(ctx, &sad->pos);
  avm_gpu_buf_destroy(ctx, &sad->cost);
  if (sad->pool) vkDestroyDescriptorPool(dev, sad->pool, NULL);
  if (sad->pipeline) vkDestroyPipeline(dev, sad->pipeline, NULL);
  if (sad->layout) vkDestroyPipelineLayout(dev, sad->layout, NULL);
  if (sad->dsl) vkDestroyDescriptorSetLayout(dev, sad->dsl, NULL);
  if (sad->module) vkDestroyShaderModule(dev, sad->module, NULL);
  free(sad);
}

/* M0 uploads one 16-bit sample per 32-bit word; see avm_gpu/README.md. */
static uint32_t *widen(const uint16_t *plane, int stride, int rows) {
  if (!plane || stride <= 0 || rows <= 0) return NULL;
  const size_t n = (size_t)stride * (size_t)rows;
  uint32_t *w = malloc(n * sizeof(*w));
  if (!w) return NULL;
  for (size_t i = 0; i < n; i++) w[i] = plane[i];
  return w;
}

int avm_gpu_sad_set_planes(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                           const uint16_t *src, int src_stride, int src_rows,
                           const uint16_t *ref, int ref_stride, int ref_rows,
                           char err[AVM_GPU_ERR_LEN]) {
  uint32_t *sw = widen(src, src_stride, src_rows);
  uint32_t *rw = widen(ref, ref_stride, ref_rows);
  if (!sw || !rw) {
    free(sw);
    free(rw);
    snprintf(err, AVM_GPU_ERR_LEN, "out of memory");
    return -1;
  }
  const VkDeviceSize src_bytes = (VkDeviceSize)src_stride * src_rows * 4;
  const VkDeviceSize ref_bytes = (VkDeviceSize)ref_stride * ref_rows * 4;
  if (ensure_buf(ctx, &sad->src, &sad->src_cap, src_bytes, err) ||
      ensure_buf(ctx, &sad->ref, &sad->ref_cap, ref_bytes, err)) {
    free(sw);
    free(rw);
    return -1;
  }
  memcpy(sad->src.mapped, sw, (size_t)src_bytes);
  memcpy(sad->ref.mapped, rw, (size_t)ref_bytes);
  free(sw);
  free(rw);
  sad->src_stride = src_stride;
  sad->ref_stride = ref_stride;
  return 0;
}

int avm_gpu_sad_run(avm_gpu_ctx *ctx, avm_gpu_sad *sad,
                    const avm_gpu_sad_pos *blocks, int n, int bw, int bh,
                    uint32_t *cost, char err[AVM_GPU_ERR_LEN]) {
  if (n <= 0) return 0;
  if (sad->src_stride <= 0 || sad->ref_stride <= 0) {
    snprintf(err, AVM_GPU_ERR_LEN, "planes not set");
    return -1;
  }
  const VkDevice dev = avm_gpu_device(ctx);
  const VkDeviceSize pos_bytes = (VkDeviceSize)n * sizeof(*blocks);
  const VkDeviceSize cost_bytes = (VkDeviceSize)n * sizeof(*cost);
  if (ensure_buf(ctx, &sad->pos, &sad->pos_cap, pos_bytes, err) ||
      ensure_buf(ctx, &sad->cost, &sad->cost_cap, cost_bytes, err)) {
    return -1;
  }
  memcpy(sad->pos.mapped, blocks, (size_t)pos_bytes);

  VkDescriptorBufferInfo info[4] = { { sad->src.buffer, 0, VK_WHOLE_SIZE },
                                     { sad->ref.buffer, 0, VK_WHOLE_SIZE },
                                     { sad->pos.buffer, 0, VK_WHOLE_SIZE },
                                     { sad->cost.buffer, 0, VK_WHOLE_SIZE } };
  VkWriteDescriptorSet wr[4];
  for (int i = 0; i < 4; i++) {
    memset(&wr[i], 0, sizeof(wr[i]));
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = sad->ds;
    wr[i].dstBinding = (uint32_t)i;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &info[i];
  }
  vkUpdateDescriptorSets(dev, 4, wr, 0, NULL);

  /* One workgroup per candidate: the whole batch rides one dispatch, which is
   * the point of pricing 128 candidates at once. */
  VkCommandBuffer cb = sad->cb;
  vkResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bi);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, sad->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, sad->layout, 0, 1,
                          &sad->ds, 0, NULL);
  push_constants pc = { (uint32_t)bw, (uint32_t)bh, (uint32_t)sad->src_stride,
                        (uint32_t)sad->ref_stride, (uint32_t)n };
  vkCmdPushConstants(cb, sad->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                     &pc);
  vkCmdDispatch(cb, (uint32_t)n, 1, 1);
  vkEndCommandBuffer(cb);

  const int rc = avm_gpu_submit_wait(ctx, cb, err);
  if (rc == 0) memcpy(cost, sad->cost.mapped, (size_t)cost_bytes);
  return rc;
}
