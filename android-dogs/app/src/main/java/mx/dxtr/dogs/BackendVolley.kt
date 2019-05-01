package mx.dxtr.dogs

import android.app.Application
import android.text.TextUtils
import com.android.volley.Request
import com.android.volley.RequestQueue
import com.android.volley.toolbox.Volley
import com.android.volley.toolbox.ImageLoader



class BackendVolley : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    private var _requestQueue: RequestQueue? = null
    private var mImageLoader: ImageLoader? = null
    val requestQueue: RequestQueue?
        get() {
            if (_requestQueue == null) {
                _requestQueue = Volley.newRequestQueue(applicationContext)
            }
            return _requestQueue
        }

    fun <T> addToRequestQueue(request: Request<T>, tag: String) {
        request.tag = if (TextUtils.isEmpty(tag)) TAG else tag
        requestQueue?.add(request)
    }

    fun cancelPendingRequests(tag: Any) {
        if (requestQueue != null) {
            requestQueue!!.cancelAll(tag)
        }
    }

    fun getImageLoader(): ImageLoader? {
        if (mImageLoader == null) {
            this.mImageLoader = ImageLoader(this.requestQueue, LruBitmapCache())
        }
        return this.mImageLoader
    }

    companion object {
        private val TAG = BackendVolley::class.java.simpleName
        @get:Synchronized var instance: BackendVolley? = null
            private set
    }
}