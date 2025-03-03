#ifndef LLVM_PROFILEDATA_MEMPROFDATA_INC
#define LLVM_PROFILEDATA_MEMPROFDATA_INC
/*===-- MemProfData.inc - MemProf profiling runtime structures -*- C++ -*-=== *\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/
/*
 * This is the main file that defines all the data structure, signature,
 * constant literals that are shared across profiling runtime library,
 * and host tools (reader/writer).
 *
 * This file has two identical copies. The primary copy lives in LLVM and
 * the other one sits in compiler-rt/include/profile directory. To make changes
 * in this file, first modify the primary copy and copy it over to compiler-rt.
 * Testing of any change in this file can start only after the two copies are
 * synced up.
 *
\*===----------------------------------------------------------------------===*/

#ifdef _MSC_VER
#define PACKED(...) __pragma(pack(push,1)) __VA_ARGS__ __pragma(pack(pop))
#else
#define PACKED(...) __VA_ARGS__ __attribute__((__packed__))
#endif

// A 64-bit magic number to uniquely identify the raw binary memprof profile file.
#define MEMPROF_RAW_MAGIC_64                                                                        \
  ((uint64_t)255 << 56 | (uint64_t)'m' << 48 | (uint64_t)'p' << 40 | (uint64_t)'r' << 32 |          \
   (uint64_t)'o' << 24 | (uint64_t)'f' << 16 | (uint64_t)'r' << 8 | (uint64_t)129)

// The version number of the raw binary format.
#define MEMPROF_RAW_VERSION 1ULL

namespace llvm {
namespace memprof {
// A struct describing the header used for the raw binary memprof profile format.
PACKED(struct Header {
  uint64_t Magic;
  uint64_t Version;
  uint64_t TotalSize;
  uint64_t SegmentOffset;
  uint64_t MIBOffset;
  uint64_t StackOffset;
});


// A struct describing the information necessary to describe a /proc/maps
// segment entry for a particular binary/library identified by its build id.
PACKED(struct SegmentEntry {
  uint64_t Start;
  uint64_t End;
  uint64_t Offset;
  // This field is unused until sanitizer procmaps support for build ids for
  // Linux-Elf is implemented.
  uint8_t BuildId[32] = {0};

  SegmentEntry(uint64_t S, uint64_t E, uint64_t O) :
    Start(S), End(E), Offset(O) {}

  SegmentEntry(const SegmentEntry& S) {
    Start = S.Start;
    End = S.End;
    Offset = S.Offset;
  }

  SegmentEntry& operator=(const SegmentEntry& S) {
    Start = S.Start;
    End = S.End;
    Offset = S.Offset;
    return *this;
  }

  bool operator==(const SegmentEntry& S) const {
    return Start == S.Start &&
           End == S.End &&
           Offset == S.Offset;
  }
});

// Packed struct definition for MSVC. We can't use the PACKED macro defined in
// MemProfData.inc since it would mean we are embedding a directive (the
// #include for MIBEntryDef) into the macros which is undefined behaviour.
#ifdef _MSC_VER
__pragma(pack(push,1))
#endif

// A struct representing the heap allocation characteristics of a particular
// runtime context. This struct is shared between the compiler-rt runtime and
// the raw profile reader. The indexed format uses a separate, self-describing
// backwards compatible format.
struct MemInfoBlock{

#define MIBEntryDef(NameTag, Name, Type) Type Name;
#include "MIBEntryDef.inc"
#undef MIBEntryDef

bool operator==(const MemInfoBlock& Other) const {
  bool IsEqual = true;
#define MIBEntryDef(NameTag, Name, Type) \
  IsEqual = (IsEqual && Name == Other.Name);
#include "MIBEntryDef.inc"
#undef MIBEntryDef
  return IsEqual;
}

MemInfoBlock() : AllocCount(0) {}

MemInfoBlock(uint32_t size, uint64_t access_count, uint32_t alloc_timestamp,
             uint32_t dealloc_timestamp, uint32_t alloc_cpu, uint32_t dealloc_cpu)
    : AllocCount(1), TotalAccessCount(access_count),
      MinAccessCount(access_count), MaxAccessCount(access_count),
      TotalSize(size), MinSize(size), MaxSize(size),
      AllocTimestamp(alloc_timestamp), DeallocTimestamp(dealloc_timestamp),
      TotalLifetime(dealloc_timestamp - alloc_timestamp),
      MinLifetime(TotalLifetime), MaxLifetime(TotalLifetime),
      AllocCpuId(alloc_cpu), DeallocCpuId(dealloc_cpu),
      NumLifetimeOverlaps(0), NumSameAllocCpu(0),
      NumSameDeallocCpu(0) {
  NumMigratedCpu = AllocCpuId != DeallocCpuId;
}

void Merge(const MemInfoBlock &newMIB) {
  AllocCount += newMIB.AllocCount;

  TotalAccessCount += newMIB.TotalAccessCount;
  MinAccessCount = newMIB.MinAccessCount < MinAccessCount ? newMIB.MinAccessCount : MinAccessCount;
  MaxAccessCount = newMIB.MaxAccessCount < MaxAccessCount ? newMIB.MaxAccessCount : MaxAccessCount;

  TotalSize += newMIB.TotalSize;
  MinSize = newMIB.MinSize < MinSize ? newMIB.MinSize : MinSize;
  MaxSize = newMIB.MaxSize < MaxSize ? newMIB.MaxSize : MaxSize;

  TotalLifetime += newMIB.TotalLifetime;
  MinLifetime = newMIB.MinLifetime < MinLifetime ? newMIB.MinLifetime : MinLifetime;
  MaxLifetime = newMIB.MaxLifetime > MaxLifetime ? newMIB.MaxLifetime : MaxLifetime;

  // We know newMIB was deallocated later, so just need to check if it was
  // allocated before last one deallocated.
  NumLifetimeOverlaps += newMIB.AllocTimestamp < DeallocTimestamp;
  AllocTimestamp = newMIB.AllocTimestamp;
  DeallocTimestamp = newMIB.DeallocTimestamp;

  NumSameAllocCpu += AllocCpuId == newMIB.AllocCpuId;
  NumSameDeallocCpu += DeallocCpuId == newMIB.DeallocCpuId;
  AllocCpuId = newMIB.AllocCpuId;
  DeallocCpuId = newMIB.DeallocCpuId;
}

#ifdef _MSC_VER
} __pragma(pack(pop));
#else
} __attribute__((__packed__));
#endif

} // namespace memprof
} // namespace llvm

#endif
