package mx.dxtr.dogs

import android.Manifest
import android.content.Intent
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import com.android.volley.VolleyError
import com.android.volley.toolbox.ImageLoader
import es.dmoral.toasty.Toasty
import org.json.JSONObject
import java.util.*
import android.app.DatePickerDialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.OpenableColumns
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.widget.*
import java.io.*
import java.text.SimpleDateFormat
import java.util.Locale

class PhotoActivity : AppCompatActivity() {

    lateinit var picture : ImageView
    lateinit var prediction : TextView
    lateinit var bitmap : Bitmap
    lateinit var bytes : ByteArray
    val service = ServiceVolley()
    var nombre = ""
    private val MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE = 1


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_photo)

        picture = findViewById(R.id.photo)
        prediction = findViewById(R.id.prediction)

        val imageLoader = BackendVolley.instance?.getImageLoader()
        val path = "api_profiles/update_profile/"
        val params = JSONObject()
        val headers = HashMap<String, String>()
    }

    fun uploadPicture(){

        val path = "http://192.168.1.72:5000/"
        val params = HashMap<String,String>()
        val headers = HashMap<String, String>()
        val param = "file"
        Toasty.info(applicationContext,"Uploading photo, wait for prediction", Toast.LENGTH_SHORT).show()
        service.post_image(path, params, headers, param, nombre, bytes) { response ->
            val responseHeaders = response?.headers
            if(responseHeaders?.get("code") == "200"){
                prediction.text = responseHeaders["prediction"]
            }
        }
    }

    fun requestRead(v:View){
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Log.d("PERMISOS","NO HAY")
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE)
        }
        selectImage()
    }

    fun selectImage(){
        val ringIntent = Intent()
        ringIntent.type = "image/*"
        ringIntent.action = Intent.ACTION_GET_CONTENT
        ringIntent.addCategory(Intent.CATEGORY_OPENABLE)
        startActivityForResult(Intent.createChooser(ringIntent, "Select Profile Picture"), 111)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 111 && resultCode == RESULT_OK) {
            val selectedFile = data?.data //The uri with the location of the file
            updatePicture(selectedFile)
        }
    }

    fun getFileName(uri: Uri?): String {
        var result: String? = null
        if (uri?.scheme == "content") {
            val cursor = contentResolver.query(uri, null, null, null, null)
            try {
                if (cursor != null && cursor.moveToFirst()) {
                    result = cursor.getString(cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME))
                }
            } finally {
                cursor!!.close()
            }
        }
        if (result == null) {
            result = uri?.path
            val cut = result!!.lastIndexOf('/')
            if (cut != -1) {
                result = result.substring(cut + 1)
            }
        }
        return result
    }

    fun updatePicture(uri : Uri?){
        val buf = BufferedInputStream(contentResolver.openInputStream(uri))
        bytes = getBytes(buf)
        nombre = getFileName(uri)
        picture.setImageBitmap(BitmapFactory.decodeByteArray(bytes, 0, bytes.size))
        uploadPicture()
    }

    @Throws(IOException::class)
    fun getBytes(inputStream: InputStream): ByteArray {
        val byteBuffer = ByteArrayOutputStream()
        val bufferSize = 1024
        val buffer = ByteArray(bufferSize)

        var len = 0
        len = inputStream.read(buffer)
        while ((len) != -1) {
            byteBuffer.write(buffer, 0, len)
            len = inputStream.read(buffer)
        }
        return byteBuffer.toByteArray()
    }

}
