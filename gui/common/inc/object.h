#define OBJECT_ALLOC(obj) \
    obj##_t* obj = (obj##_t*)calloc(1, sizeof(obj##_t));
#define OBJECT_DELETE_MEMBER(parent, obj) \
    if (parent->obj) { obj##_delete(parent->obj); }
