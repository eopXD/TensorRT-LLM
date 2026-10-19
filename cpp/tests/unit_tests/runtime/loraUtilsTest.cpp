/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include "tensorrt_llm/common/tllmDataType.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tensorrt_llm::runtime::lora
{
using TensorPtr = ITensor::SharedPtr;

class LoraUtilsTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    LoraUtilsTest() {}

    void SetUp() override
    {
        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);
    }

    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
};

TEST_F(LoraUtilsTest, null_values)
{
    std::optional<TensorPtr> optReqLoraWeights = std::nullopt;
    std::optional<TensorPtr> optReqLoraConfig = mManager->emptyTensor(MemoryType::kCPU, tensorrt_llm::DataType::kHALF);

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());

    optReqLoraConfig = std::nullopt;

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());
}

TEST_F(LoraUtilsTest, dims_mem_type)
{
    std::optional<TensorPtr> optReqLoraWeights
        = mManager->cpu(ITensor::makeShape({1, 2}), tensorrt_llm::DataType::kHALF);
    std::optional<TensorPtr> optReqLoraConfig
        = mManager->cpu(ITensor::makeShape({1, 2, 3}), tensorrt_llm::DataType::kINT32);

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());

    std::optional<TensorPtr> optGpuWeights
        = mManager->gpu(ITensor::makeShape({1, 2, 50}), tensorrt_llm::DataType::kHALF);

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optGpuWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());

    optReqLoraWeights = mManager->cpu(ITensor::makeShape({1, 2, 50}), tensorrt_llm::DataType::kHALF);
    optReqLoraConfig = mManager->cpu(ITensor::makeShape({1, 2, 3}), tensorrt_llm::DataType::kINT32);

    loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig);
}

TEST_F(LoraUtilsTest, loraValidateRequestTensors)
{
    auto modelConfig = ModelConfig(0, 2, 2, 0, 1, 4, tensorrt_llm::DataType::kFLOAT);
    auto worldConfig = WorldConfig();

    std::optional<TensorPtr> optReqLoraWeights
        = mManager->cpu(ITensor::makeShape({1, 2, 32}), tensorrt_llm::DataType::kFLOAT);
    std::optional<TensorPtr> optReqLoraConfig
        = mManager->cpu(ITensor::makeShape({1, 2, 3}), tensorrt_llm::DataType::kINT32);

    std::vector<int32_t> config{1, 0, 4, 1, 1, 4};

    auto configPtr = optReqLoraConfig.value()->getBufferPtr<int32_t>();
    std::copy_n(config.data(), config.size(), configPtr);

    EXPECT_THAT([&]()
        { loraValidateRequestTensors(12345, optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig); },
        testing::Throws<std::runtime_error>());

    std::vector<LoraModule> modules{
        LoraModule(LoraModule::ModuleType::kATTN_Q, 4, 4, false, true, -1, 0),
    };
    modelConfig.setLoraModules(modules);
    EXPECT_THAT([&]()
        { loraValidateRequestTensors(12345, optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig); },
        testing::Throws<std::runtime_error>());

    modelConfig.setMaxLoraRank(4);

    loraValidateRequestTensors(12345, optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig);

    EXPECT_THAT([&]()
        { loraValidateRequestTensors(std::nullopt, optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig); },
        testing::Throws<std::runtime_error>());
}

TEST(LoraModuleTest, executionIdDefaultsToSemanticType)
{
    constexpr LoraModule decoder(LoraModule::ModuleType::kATTN_Q, 4, 4, false, true, -1, -1);
    static_assert(decoder.executionId() == static_cast<SizeType32>(LoraModule::ModuleType::kATTN_Q));
    EXPECT_EQ(decoder.value(), decoder.executionId());

    constexpr SizeType32 kComponentExecutionId = 26;
    constexpr LoraModule component(LoraModule::ModuleType::kATTN_Q, 8, 12, false, true, -1, -1, kComponentExecutionId);
    static_assert(component.executionId() == kComponentExecutionId);
    EXPECT_EQ(component.value(), decoder.value());
    EXPECT_EQ(component.name(), decoder.name());
    EXPECT_EQ(component.flattenedInOutSize(2, false), 40);
}

TEST(LoraModuleTest, executionIdRejectsNegativeValues)
{
    for (auto const executionId : {-1, std::numeric_limits<SizeType32>::min()})
    {
        SCOPED_TRACE(executionId);
        EXPECT_THROW(
            LoraModule(LoraModule::ModuleType::kATTN_Q, 4, 4, false, true, -1, -1, executionId), std::runtime_error);
    }
}

TEST_F(LoraUtilsTest, executionIdsSelectIndependentProjectionDimensions)
{
    ModelConfig modelConfig(0, 1, 1, 0, 1, 4, tensorrt_llm::DataType::kFLOAT);
    WorldConfig worldConfig(1, 1, 1, 0);
    modelConfig.setMaxLoraRank(2);
    modelConfig.setLoraModules({
        LoraModule(LoraModule::ModuleType::kATTN_Q, 4, 4, false, true, -1, -1),
        LoraModule(LoraModule::ModuleType::kATTN_Q, 8, 12, false, true, -1, -1, 26),
    });
    std::vector<int32_t> rows{1, 0, 2, 0, 26, 0, 2, 0};
    TensorPtr config = ITensor::wrap(rows, ITensor::makeShape({1, 2, 4}));
    TensorPtr weights = mManager->cpu(ITensor::makeShape({1, 2, 40}), tensorrt_llm::DataType::kFLOAT);

    EXPECT_NO_THROW(loraValidateRequestTensors(7, weights, config, modelConfig, worldConfig));

    weights = mManager->cpu(ITensor::makeShape({1, 2, 16}), tensorrt_llm::DataType::kFLOAT);
    EXPECT_THROW(loraValidateRequestTensors(7, weights, config, modelConfig, worldConfig), std::runtime_error);
}

TEST_F(LoraUtilsTest, executionIdsRejectInvalidRequestsAndDuplicateDestinations)
{
    ModelConfig modelConfig(0, 2, 2, 0, 1, 4, tensorrt_llm::DataType::kFLOAT);
    WorldConfig worldConfig(1, 1, 1, 0);
    modelConfig.setMaxLoraRank(2);
    modelConfig.setLoraModules({LoraModule(LoraModule::ModuleType::kATTN_Q, 4, 4, false, true, -1, -1, 26)});
    std::vector<int32_t> rows{26, 0, 2, 0, 26, 1, 2, 0};
    TensorPtr config = ITensor::wrap(rows, ITensor::makeShape({1, 2, 4}));
    TensorPtr weights = mManager->cpu(ITensor::makeShape({1, 2, 16}), tensorrt_llm::DataType::kFLOAT);

    EXPECT_NO_THROW(loraValidateRequestTensors(7, weights, config, modelConfig, worldConfig));
    rows[5] = 0;
    EXPECT_THROW(loraValidateRequestTensors(7, weights, config, modelConfig, worldConfig), std::runtime_error);
    rows[5] = 1;
    for (auto const executionId : {-1, 1, 27})
    {
        SCOPED_TRACE(executionId);
        rows[0] = executionId;
        EXPECT_THROW(loraValidateRequestTensors(7, weights, config, modelConfig, worldConfig), std::runtime_error);
    }
}

TEST_F(LoraUtilsTest, executionIdsRejectDuplicateModelDeclarations)
{
    ModelConfig modelConfig(0, 1, 1, 0, 1, 4, tensorrt_llm::DataType::kFLOAT);
    WorldConfig worldConfig(1, 1, 1, 0);
    modelConfig.setMaxLoraRank(2);
    modelConfig.setLoraModules({
        LoraModule(LoraModule::ModuleType::kATTN_Q, 4, 4, false, true, -1, -1, 26),
        LoraModule(LoraModule::ModuleType::kATTN_K, 4, 4, false, true, -1, -1, 26),
    });
    std::vector<int32_t> rows{26, 0, 2, 0};
    TensorPtr config = ITensor::wrap(rows, ITensor::makeShape({1, 1, 4}));
    TensorPtr weights = mManager->cpu(ITensor::makeShape({1, 1, 16}), tensorrt_llm::DataType::kFLOAT);

    EXPECT_THROW(loraValidateRequestTensors(7, weights, config, modelConfig, worldConfig), std::runtime_error);
}

} // namespace tensorrt_llm::runtime::lora
