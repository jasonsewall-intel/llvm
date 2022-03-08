#include <CL/sycl.hpp>

#include <cstdio>

namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

  struct temporality_hint_key {
    template <typename T>
    using value_t = property_value<temporality_hint_key, T>;
  };

  struct nontemporal { static constexpr int hint = 0x4; };
  struct temporal { static constexpr int hint = 0x0; };

  struct L1_cache_hint_key {
    template <typename T>
    using value_t = property_value<L1_cache_hint_key, T>;
  };

  struct L2_cache_hint_key {
    template <typename T>
    using value_t = property_value<L2_cache_hint_key, T>;
  };

  struct L3_cache_hint_key  {
    template <typename T>
    using value_t = property_value<L3_cache_hint_key, T>;
  };

  struct L4_cache_hint_key {
    template <typename T>
    using value_t = property_value<L4_cache_hint_key, T>;
  };

  template <> struct is_property_key<temporality_hint_key> : std::true_type {};
  template <> struct is_property_key<L1_cache_hint_key> : std::true_type {};
  template <> struct is_property_key<L2_cache_hint_key> : std::true_type {};
  template <> struct is_property_key<L3_cache_hint_key> : std::true_type {};
  template <> struct is_property_key<L4_cache_hint_key> : std::true_type {};

  template <> struct detail::IsCompileTimeProperty<temporality_hint_key> : std::true_type {};
  template <> struct detail::IsCompileTimeProperty<L1_cache_hint_key> : std::true_type {};
  template <> struct detail::IsCompileTimeProperty<L2_cache_hint_key> : std::true_type {};
  template <> struct detail::IsCompileTimeProperty<L3_cache_hint_key> : std::true_type {};
  template <> struct detail::IsCompileTimeProperty<L4_cache_hint_key> : std::true_type {};

  namespace detail {
    template <> struct PropertyToKind<temporality_hint_key> {
      static constexpr PropKind Kind =
        static_cast<enum PropKind>(PropKind::PropKindSize + 0);
    };
    template <> struct PropertyToKind<L1_cache_hint_key> {
      static constexpr PropKind Kind =
        static_cast<enum PropKind>(PropKind::PropKindSize + 1);
    };
    template <> struct PropertyToKind<L2_cache_hint_key> {
      static constexpr PropKind Kind =
        static_cast<enum PropKind>(PropKind::PropKindSize + 2);
    };
    template <> struct PropertyToKind<L3_cache_hint_key> {
      static constexpr PropKind Kind =
        static_cast<enum PropKind>(PropKind::PropKindSize + 3);
    };
    template <> struct PropertyToKind<L4_cache_hint_key> {
      static constexpr PropKind Kind =
        static_cast<enum PropKind>(PropKind::PropKindSize + 4);
    };
  }

  template <typename T>
  inline constexpr temporality_hint_key::value_t<T> temporality_hint;

  template <typename T>
  inline constexpr L1_cache_hint_key::value_t<T> L1_cache_hint;
  template <typename T>
  inline constexpr L2_cache_hint_key::value_t<T> L2_cache_hint;
  template <typename T>
  inline constexpr L3_cache_hint_key::value_t<T> L3_cache_hint;
  template <typename T>
  inline constexpr L4_cache_hint_key::value_t<T> L4_cache_hint;

  using EmptyP = decltype(properties());

  template <typename T, typename Props=EmptyP>
  T load(const T *addr) {
    return load<T,Props>(addr, Props());
  }

  template <typename T, typename Props>
  T load(const T *addr, Props p) {
    //static_assert(!p.has_runtime_property());

    if constexpr (Props::template has_property<temporality_hint_key>()) {
      constexpr auto prop = Props::template get_property<temporality_hint_key>();
      using vt= typename decltype(prop)::value_t;
      ::printf("__spirv_load(%p, 0x%x);\n", addr, vt::hint);
    }
    else {
      ::printf("__spirv_load(%p, 0x0);//default temporal load (nothing passed)\n", addr);
    }
    return *addr;
  }

  template <typename T, typename Props>
  void store(T *addr, T &value, Props p) {
    //static_assert(!p.has_runtime_property());

    if constexpr (Props::template has_property<temporality_hint_key>()) {
      constexpr auto prop = Props::template get_property<temporality_hint_key>();
      using vt = typename decltype(prop)::value_t;
      ::printf("__spirv_store(%p, val, 0x%x);\n", addr, vt::hint);
    }
    else {
      ::printf("__spirv_store(%p, val, 0x0);//default temporal store (nothing passed)\n", addr);
    }
    *addr = value;
  }

  template <typename T, typename Props=EmptyP>
  void store(T *addr, T &value) {
    store<T,Props>(addr, value, Props());
  }

  template <typename T, typename Props>
  struct annotated_ref_wrapper {
    annotated_ref_wrapper(T *datum) : datum(datum) {}

    operator T() {
      return load<T,Props>(datum);
    }

    void operator=(const T& o) {
      store<T,Props>(datum, o);
    }

    void operator=(T& o) {
      store<T,Props>(datum, o);
    }


  private:
    T *datum;
  };

//FIXME: Const handling?
//FIXME: force compile-time only
  template <typename T, typename Props>
  struct annotated_ptr {
    annotated_ptr(T *datum, Props) : datum(datum) {}

    annotated_ref_wrapper<T, Props> operator*()  {
      return annotated_ref_wrapper<T, Props>(datum);
    }

    annotated_ptr<T, Props> operator+(unsigned long long offs) {
      return annotated_ptr<T, Props>(datum+offs);
    }

  private:
    annotated_ptr(T *datum) : datum(datum) {}
    T *datum;
  };
}}}}

using namespace sycl;

void foo() {

  double* base = new double[20];

  double d000 = ext::oneapi::experimental::load(base);

  double d0 = ext::oneapi::experimental::load(base, ext::oneapi::experimental::properties{ext::oneapi::experimental::temporality_hint<ext::oneapi::experimental::nontemporal>});
  double d1 = ext::oneapi::experimental::load(base, ext::oneapi::experimental::properties{ext::oneapi::experimental::temporality_hint<ext::oneapi::experimental::temporal>});

  auto dp = ext::oneapi::experimental::annotated_ptr(base, ext::oneapi::experimental::properties{ext::oneapi::experimental::temporality_hint<ext::oneapi::experimental::nontemporal>});
  double d5 = *dp;
  double d6 = *(dp+4);
  *dp = d1;

  delete[] base;
}

int main() {
  foo();
  return 0;
}
