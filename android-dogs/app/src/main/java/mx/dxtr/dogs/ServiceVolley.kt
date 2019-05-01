package mx.dxtr.dogs

import android.util.Log
import com.android.volley.AuthFailureError
import com.android.volley.NetworkResponse
import com.android.volley.Response
import com.android.volley.toolbox.JsonArrayRequest
import com.android.volley.toolbox.JsonObjectRequest
import com.android.volley.toolbox.JsonRequest
import org.json.JSONArray
import org.json.JSONObject
import com.android.volley.ParseError
import org.json.JSONException
import com.android.volley.toolbox.HttpHeaderParser
import java.io.UnsupportedEncodingException
import java.nio.charset.Charset
import android.R.attr.data
import android.R.attr.data






class ServiceVolley {
    val TAG = ServiceVolley::class.java.simpleName

    fun post_image(path: String, params: HashMap<String, String>, headers: HashMap<String, String>, param: String, nombre: String, bytes: ByteArray, completionHandler: (response: NetworkResponse?) -> Unit) {
        val multipartRequest = object : VolleyMultipartRequest(Method.POST, path,
                Response.Listener<NetworkResponse> { response ->
                    Log.d(TAG, "/post request OK! Response: $response")
                    val resultResponse = String(response.data)
                    val result = JSONObject(resultResponse)
                    val prediction = result.getString("prediction")
                    val responseHeaders = response.headers
                    responseHeaders["code"] = "200"
                    responseHeaders["prediction"] = prediction
                    completionHandler(response)
                },
                Response.ErrorListener { error ->
                    Log.d(TAG, "/post request fail! Error: ${error.message}")
                    completionHandler(null)
                }) {
            //pass String parameters here
            override fun getParams(): Map<String, String> {
                return params
            }

            @Throws(AuthFailureError::class)
            override fun getHeaders(): Map<String, String> {
                headers.put("Content-Type", "application/json charset=utf-8")
                return headers
            }

            override fun getByteData(): Map<String, VolleyMultipartRequest.DataPart>? {
                val params = HashMap<String, VolleyMultipartRequest.DataPart>()
                params[param] = DataPart(nombre, bytes, "image/*")
                return params
            }
        }
        BackendVolley.instance?.addToRequestQueue(multipartRequest, TAG)
    }

}